from qatch.connectors import SqliteConnector
from qatch.generate_dataset import OrchestratorGenerator


def utils_run_qatch(
    sqlite_connector: SqliteConnector, selected_col: str, tbl_name: str
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
    qatch_generator = OrchestratorGenerator(
        generator_names=[
            "project",
            "distinct",
            "select",
            "simple",
            "orderby",
            "groupby",
            "having",
        ]
    )
    df = qatch_generator.generate_dataset(
        sqlite_connector, column_to_include=selected_col, tables_to_include=[tbl_name]
    )
    df_masked = df[
        df.apply(
            lambda row: f"`{selected_col.lower()}`" in row["query"].lower(), axis=1
        )
    ]
    # remove unnecessary test-categories
    df_masked = df_masked[
        ~df_masked.sql_tag.str.lower().str.contains(
            "join|many-to-many|project-random-col|orderby-single"
        )
    ]

    # TODO undestand if it is better to include in each generator
    # sample q element for each test-category

    df_masked = (
        df_masked.groupby("test_category")
        .apply(lambda x: x.sample(2) if len(x) > 2 else x)
        .reset_index(drop=True)
    )

    list_tests = (
        df_masked.loc[:, ["test_category", "query", "question"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    return list_tests
