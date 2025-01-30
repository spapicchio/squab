import difflib
import sqlite3

from qatch.connectors import SqliteConnector
from qatch.generate_dataset import OrchestratorGenerator as QatchOrchestrator


def utils_run_qatch(sqlite_connector: SqliteConnector, selected_col: str, tbl_name: str
                    ) -> list[dict]:
    """
    Executes query generation to produce a list of unique test case configurations
    based on the provided table name and selected column.

    Args:
        sqlite_connector (SqliteConnector): The database connector to interact
            with SQLite database.
        selected_col (str): The name of the column to include in generated queries.
        tbl_name (str): The name of the table to use in query generation.

    Returns:
        list[dict]: A list of dictionaries representing unique test configurations,
            each including 'test_category', 'query', 'question'.
    """
    qatch_generator = QatchOrchestrator(
        generator_names=['project', 'distinct', 'select', 'simple', 'orderby', 'groupby', 'having']
    )
    df = qatch_generator.generate_dataset(sqlite_connector, column_to_include=selected_col, tbl_names=[tbl_name])
    df_masked = df[df.apply(lambda row: f"`{selected_col.lower()}`" in row['query'].lower(), axis=1)]
    # remove unnecessary test-categories
    df_masked = df_masked[
        ~df_masked.sql_tag.str.lower().str.contains('join|many-to-many|project-random-col|orderby-single')
    ]

    # TODO undestand if it is better to include in each generator
    # sample q element for each test-category
    sample_fun = lambda x: x.sample(2) if len(x) > 2 else x
    df_masked = df_masked.groupby('test_category').apply(sample_fun).reset_index(drop=True)

    list_tests = df_masked.loc[:, ['test_category', 'query', 'question']].drop_duplicates().to_dict(orient='records')
    return list_tests


def utils_get_db_dump_no_insert(db_path):
    """
    Generates a database dump string containing only 'CREATE TABLE' statements. Excludes INSERT statements or
    other SQL commands, returning a string of the database schema creation statements for a SQLite database.

    Args:
        db_path (str): The path to the SQLite database file.

    Returns:
        str: A single string concatenating all 'CREATE TABLE' statements from the SQLite database dump.

    Raises:
        sqlite3.Error: If there is an issue connecting to or querying the SQLite database.
    """
    with sqlite3.connect(db_path) as conn:
        # Iterates over the dump generator and join each line into a single string
        dump_string = "\n".join([line for line in conn.iterdump() if 'create table' in line.lower()])
    return dump_string


def utils_find_closest_matches(
        target_words: list[str] | str | None,
        candidate_words: list[str]
) -> list[str]:
    """
    Find the closest matching words from a list of candidate words based on syntactic similarity.

    This function takes a list or a single target word and compares it against a list
    of candidate words, returning the best match for each target word. If no target
    word is provided, the entire candidate word list is returned as is.

    Args:
        target_words (list[str] | str | None): A list of target words, a single
            target word, or None.
        candidate_words (list[str]): A list of candidate words to compare against
            the target.

    Returns:
        list[str]: A list of candidate words that are the closest matches for
            each target word.

    """
    if target_words is None:
        return candidate_words
    if isinstance(target_words, str):
        target_words = [target_words]

    def get_best_match(target: str, candidates: list[str]) -> str:
        scores = [utils_syntactic_match(target, c) for c in candidates]
        return candidates.pop(scores.index(max(scores)))

    return [get_best_match(t, candidate_words) for t in target_words]


def utils_syntactic_match(str1: str, str2: str) -> float:
    """
    Compares two strings syntactically and returns a similarity ratio.

    Uses the SequenceMatcher from the difflib library to determine the
    similarity ratio between two strings based on their syntactic content.

    Args:
        str1 (str): The first string for comparison.
        str2 (str): The second string for comparison.

    Returns:
        float: A floating-point value between 0 and 1 representing the
        similarity ratio. A value of 1 indicates identical strings, while 0
        indicates no similarity.
    """
    return difflib.SequenceMatcher(None, str1, str2).ratio()
