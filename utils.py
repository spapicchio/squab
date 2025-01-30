import datetime
import logging
import os
import re
import sqlite3
from collections import defaultdict

import pandas as pd
import sqlalchemy
from qatch.connectors import SqliteConnector


def read_db_tbl_beaver(db_path) -> list[tuple[str, list[str]]]:
    # these are the uniformly sampled tables from beaver used to generate the dataset
    selected_tbls = ["new_FCLT_ORG_DLC_KEY",
                     "FCLT_ORG_DLC_KEY",
                     "SPACE_SUPERVISOR_USAGE",
                     "FCLT_BUILDING_HIST",
                     "CIS_HASS_ATTRIBUTE",
                     "FAC_BUILDING_ADDRESS",
                     "MASTER_DEPT_HIERARCHY",
                     "ACADEMIC_TERMS_ALL",
                     "FAC_FLOOR",
                     "LIBRARY_MATERIAL_STATUS",
                     "TIP_DETAIL",
                     "SIS_COURSE_DESCRIPTION",
                     "LIBRARY_RESERVE_CATALOG",
                     "TIME_DAY",
                     "FCLT_ORGANIZATION",
                     "FCLT_BUILDING_ADDRESS",
                     "IAP_SUBJECT_SESSION",
                     "TIP_MATERIAL_STATUS",
                     "ACADEMIC_TERM_PARAMETER",
                     "FCLT_BUILDING",
                     "SPACE_USAGE",
                     "SPACE_UNIT",
                     "TIP_MATERIAL",
                     "SPACE_FLOOR",
                     "IAP_SUBJECT_SPONSOR",
                     "IAP_SUBJECT_DETAIL",
                     "FAC_ORGANIZATION",
                     "SIS_SUBJECT_CODE",
                     "SIS_ADMIN_DEPARTMENT",
                     "STUDENT_DEPARTMENT",
                     "BUILDINGS",
                     "ACADEMIC_TERMS",
                     "SIS_DEPARTMENT",
                     "CIP"]
    tables = {tbl: pd.read_csv(os.path.join(db_path, f"{tbl}.csv")) for tbl in selected_tbls}
    SqliteConnector(relative_db_path=os.path.join(db_path, 'beaver.sqlite'), db_name='beaver', tables=tables)

    return [(os.path.join(db_path, 'beaver.sqlite'), selected_tbls)]


def read_db_tbl_amrbosia_unans(db_path):
    """
    Processes database tables from Ambrosia ambiguous database scope and retrieves schema length for each table.

    This function reads database tables under the Ambrosia ambiguous database 'scope' category. It processes each
    table by retrieving its schema information to determine the number of columns in each table. The results are
    then grouped and returned as a dictionary mapping database paths to their corresponding tables.

    Returns:
        defaultdict[str, set[str]]: A mapping of database paths to sets of table names for which schema
        information was processed and sorted by the number of columns.
    """
    scope_db_tbls = read_db_tbl_ambrosia_ambig(db_path, 'scope')
    db_tbls_len_schema = []
    for db_path, tables in scope_db_tbls:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            for table_name in tables:
                # Example query to process each table
                # Use PRAGMA to retrieve the table schema
                cursor.execute(f"PRAGMA table_info({table_name});")
                schema_info = cursor.fetchall()
                if schema_info:
                    # The length of the schema corresponds to the number of retrieved columns
                    db_tbls_len_schema.append((db_path, table_name, len(schema_info)))

    db_tbls_len_schema = sorted(db_tbls_len_schema, key=lambda x: x[-1])

    db_path2tbls = defaultdict(set)
    for db_dict in db_tbls_len_schema:
        db_path2tbls[db_dict['db_path']].add(db_dict['tbl_name'])

    return db_path2tbls


def read_db_tbl_ambrosia_ambig(path, ambig_type) -> list[tuple[str, list[str]]]:
    """
    Retrieves database table information based on the specified ambiguity type
    from the Ambrosia dataset.

    This function reads data from the Ambrosia dataset and extracts table
    information that corresponds to a specific ambiguity type. For non-scope
    ambiguity types, database paths are grouped based on the database file
    and table names. For scope ambiguity, denormalization is applied before
    retrieving database paths.

    Args:
        ambig_type (str): The type of ambiguity to filter for (e.g., 'scope').

    Returns:
        list[tuple[str, list[str]]]: A sorted list of tuples where each tuple
        contains a database file path as the first element and a list of
        associated table names as the second element.
    """
    df = pd.read_csv(path)
    # get only ambiguous tests
    df = df[df.question_type == 'ambig']
    # get specific ambiguity type
    df = df[df.ambig_type == ambig_type]
    if ambig_type != 'scope':
        df = ambrosia_only_single_tbl(df)
        df.db_file = df.db_file.map(lambda x: x.replace('data', 'data/ambrosia'))
        db_paths = set(map(tuple, df[['db_file', 'tbl_name']].values))
        output = defaultdict(list)
        for db, tbl in db_paths:
            output[db].append(tbl)
        db_paths = [(k, v) for k, v in output.items()]
    else:
        # for scope, we need to create a denormalized db
        db_paths = denormalize_and_save_ambrosia(df)
    return sorted(db_paths)


def ambrosia_only_single_tbl(ambrosia_df) -> pd.DataFrame:
    """
    Processes a DataFrame to filter and extract rows related to single tables based on SQL query content.

    This function modifies the given DataFrame by evaluating the 'ambig_queries' column to identify
    single table queries that do not involve operations like joins or unions. It further processes
    the result to extract the table name when only a single table is involved in the queries.

    Args:
        ambrosia_df (pd.DataFrame): DataFrame containing a column 'ambig_queries', where each entry
            contains a list of SQL queries in string format.

    Returns:
        pd.DataFrame: A filtered and modified DataFrame where each row corresponds to single table
            queries, and the associated table name is added in a new column 'tbl_name'.

    Raises:
        ValueError: If the 'tbl_name' column contains sets with more than one table during
            processing.
    """

    def extract_tbl(tbls: set):
        if len(tbls) != 1:
            raise ValueError(tbls)
        return tbls.pop()[0]

    ambrosia_df['ambig_queries'] = ambrosia_df['ambig_queries'].map(eval)
    ambrosia_df_no_multiple_tables = ambrosia_df[
        ambrosia_df['ambig_queries'].map(
            lambda queries: all(
                'join' not in query.lower()
                and 'union' not in query.lower()
                # and 'in' not in query.lower()
                for query in queries
            )
        )
    ]
    ambrosia_df_no_multiple_tables.loc[:, 'tbl_name'] = ambrosia_df_no_multiple_tables['ambig_queries'].map(
        lambda queries: set.union({utils_extract_tables_from_sql(query) for query in queries})
    )
    # remove tests with multiple tables
    ambrosia_df_no_multiple_tables.loc[:, 'tbl_name'] = ambrosia_df_no_multiple_tables[
        ambrosia_df_no_multiple_tables['tbl_name'].map(lambda x: len(x) == 1)
    ]
    ambrosia_df_no_multiple_tables.loc[:, 'tbl_name'] = ambrosia_df_no_multiple_tables['tbl_name'].map(extract_tbl)
    return ambrosia_df_no_multiple_tables


def denormalize_and_save_ambrosia(db_paths):
    """
    Denormalizes database tables and saves the processed data into a specified directory. Existing
    files are NOT overwritten if they exist. The function handles errors gracefully and skips problematic
    databases. The function returns paths of newly created denormalized database files along with their
    tables.

    Args:
        db_paths (set): A set of paths pointing to databases that need to be denormalized. Each path
            should be related to the original database files.

    Returns:
        list: A list of tuples where each tuple contains the path to the denormalized database and its
            corresponding tables. Returns an empty list if no databases are successfully denormalized.

    Raises:
        Exception: Exception is raised internally if there are issues in database processing,
            specifically when denormalizing tables or saving new databases. Such exceptions are
            logged and the corresponding database is skipped.
    """
    # db_paths = set(df.db_file.map(lambda x: x.replace('data', 'data/ambrosia')).to_list())

    db_paths_denormilized = []
    for db_path in db_paths:

        dp_path_denormalized = db_path.replace('data/ambrosia/', 'data/ambrosia_denormalized/')

        if os.path.exists(dp_path_denormalized):
            continue
        try:
            tables = denormalize_table_in_database(db_path)
        except Exception as e:
            logging.warning(f'{db_path}: error Generating new Table: {e}')
            continue
        else:
            if tables is not None and len(tables) > 0:
                dir = '/'.join(dp_path_denormalized.split('/')[:-1])
                if not os.path.exists(dir):
                    os.makedirs(dir)
                try:
                    _ = SqliteConnector(
                        relative_db_path=dp_path_denormalized,
                        db_name=db_path.split('/')[-1].replace('.sqlite', ''),
                        tables=tables

                    )
                except Exception as e:
                    logging.warning(f'{db_path}: error Generating new Table: {e}')
                    continue
                else:
                    db_paths_denormilized.append((dp_path_denormalized, tables))
    return db_paths_denormilized


def utils_extract_tables_from_sql(query):
    """
    Extracts unique table names from an SQL query.

    This function uses regular expressions to identify table names
    present in the `FROM` and `JOIN` clauses of a given SQL query.
    If more than one unique table name is found, a warning is logged.
    Returns a tuple containing the unique table names.

    Args:
        query (str): The SQL query to analyze and extract table names from.

    Returns:
        tuple: A tuple of unique table names extracted from the query.

    Raises:
        None: This function does not raise any exceptions explicitly.
    """
    # regex pattern for matching table names
    pattern = re.compile(
        r'\bFROM\s+([`"\']?[\w\.]+[`"\']?)|\bJOIN\s+([`"\']?[\w\.]+[`"\']?)',
        re.IGNORECASE)

    # find all matches in the query
    matches = pattern.findall(query)

    # flatten the list of tuples and remove None values
    tables = {m for match in matches for m in match if m}
    if len(tables) > 1:
        # raise ValueError(query)
        logging.warning(f'Found {tables} tables in `{query}`')

    # return unique table names
    return tuple(tables)


def denormalize_table_in_database(db_path):
    """
    Denormalizes tables in the given SQLite database by resolving foreign key relationships
    and merging tables. The resulting denormalized tables will combine related data into single
    tables. If any tables contain no data or are already denormalized, they are appropriately
    handled or skipped.

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        dict: A dictionary where keys are the names of denormalized tables and values are
        corresponding pandas DataFrames. Returns an empty dictionary if no denormalized tables
        are generated, and None if the database cannot be accessed or has no tables.

    Raises:
        sqlalchemy.exc.NoSuchTableError: If the specified table(s) do not exist in the database.

    """
    try:
        connector = SqliteConnector(db_path, db_name='_')
    except sqlalchemy.exc.NoSuchTableError as e:
        logging.warning(f'{db_path} error table does not exists')
        return None

    tables = connector.load_tables_from_database()
    tables_name_denormilized = []
    denormalized_tables = {}

    tables = {tbl_name: tbl
              for tbl_name, tbl in tables.items()
              if 'denormalized' not in tbl_name and 'sqlite' not in tbl_name
              }

    for tbl_name, table in tables.items():

        table_already_seen = {tbl_name}

        table1 = _read_table_date_format(tbl_name, connector.engine)

        foreign_keys_tbl = {(val['parent_column'], val['child_column'], val['child_table'].tbl_name)
                            for val in table.foreign_keys}

        if len(table1) == 0:
            continue

        while foreign_keys_tbl:
            foreign_key = foreign_keys_tbl.pop()
            if foreign_key[2] not in table_already_seen:
                table_already_seen.add(foreign_key[2])
                table2 = _read_table_date_format(foreign_key[2], connector.engine)
                table1 = pd.merge(table1,
                                  table2,
                                  how="inner",
                                  left_on=foreign_key[0],
                                  right_on=foreign_key[1],
                                  suffixes=(f'_{table.tbl_name}', f'_{foreign_key[2]}')
                                  )
                foreign_keys_tbl = foreign_keys_tbl.union({
                    (val['parent_column'], val['child_column'], val['child_table'].tbl_name)
                    for val in tables[foreign_key[2]].foreign_keys if
                    val['child_table'].tbl_name not in table_already_seen
                })

        if len(table_already_seen) > 1:
            # remove duplicate columns in table1
            table1.columns = [c.lower() for c in table1.columns]

            table1 = table1.loc[:, ~table1.columns.duplicated()]
            denormalized_tables['_'.join(table_already_seen)] = table1
            tables_name_denormilized += list(table_already_seen)

    # at this point
    for tbl_name, table in tables.items():
        if tbl_name not in tables_name_denormilized and 'denormalized' not in tbl_name and 'sqlite' not in tbl_name:
            table1 = _read_table_date_format(tbl_name, connector.engine)
            denormalized_tables[tbl_name] = table1

    return denormalized_tables


def str_to_isoformat(dt_str):
    """
    Converts a datetime string in a specific format to ISO 8601 format. If the input
    is invalid or does not match the expected format, returns the original string.

    Args:
        dt_str (str): A string representing a datetime in the format
            '%Y-%m-%d %I:%M:%S %p'.

    Returns:
        str: The ISO 8601 formatted datetime string if the input is valid; otherwise,
            the original input string.
    """
    try:
        return datetime.datetime.strptime(dt_str, '%Y-%m-%d %I:%M:%S %p').isoformat()
    except (ValueError, TypeError):
        return dt_str


def _read_table_date_format(tbl_name, engine):
    """
    Reads a database table and applies date format conversion to columns with string/object data type, ensuring
    they are converted to an ISO 8601 formatted string if applicable.

    Args:
        tbl_name (str): The name of the table to read from the database.
        engine (sqlalchemy.engine.Engine): SQLAlchemy database engine to
            connect and execute the query.

    Returns:
        pandas.DataFrame: The DataFrame containing the table data with formatted
            date columns, where applicable.
    """
    table1 = pd.read_sql_query(f"SELECT * FROM {tbl_name}", con=engine)
    for column in table1.columns:
        # If the column dtype is an object, it could be a string.
        if table1[column].dtype == 'object':
            # Try to apply conversion to ISO formatted string.
            table1[column] = table1[column].apply(str_to_isoformat)
    return table1
