import json
import re
from enum import Enum
from typing import TypeVar

from qatch.connectors import SqliteConnector
from qatch.generate_dataset import OrchestratorGenerator
from rapidfuzz.distance import Levenshtein
from typing_extensions import Literal

from squab.graph_states import Line

T = TypeVar('T')


class GenerationSteps(Enum):
    PI = 'pattern_identification'
    RM = 'relational_metadata'
    TG = 'test_generation'


def utils_get_columns_no_pk_fk(line: Line,
                               start_from_cols: list[str] | None = None
                               ) -> list[str]:
    """
    Retrieves a list of column names excluding the primary key, foreign key, and certain unwanted patterns.

    The function processes the provided or default column names of a table and excludes
    those that are primary keys, foreign keys, or contain specific substrings such as
    `id`, `code`, or `key` in their name.

    Args:
        ...

    Returns:
        list[str]: A filtered list of column names, excluding primary keys, foreign keys,
        and those containing `id`, `code`, or `key` substrings.
    """
    column_names = start_from_cols or list(line["tbl_col2metadata"].keys())
    primary_keys_name = [pk["column_name"] for pk in line["primary_key"]] if line["primary_key"] else []
    primary_keys_name += [fk['parent_column'] for fk in line["foreign_keys"]] if line["foreign_keys"] else []
    column_names = [val for val in column_names if
                    val not in primary_keys_name and
                    'id' not in val.lower() and
                    'code' not in val.lower() and
                    'key' not in val.lower()]
    return column_names


def utils_get_last_json_from_text(text: str) -> dict:
    """
    Extracts the last JSON object from a given text string.

    Args:
        text (str): The input text containing JSON objects.

    Returns:
        dict: The last JSON object found in the text.
    """
    # Find all JSON-like patterns in the text
    json_objects = utils_get_json_blocks_from_text(text)
    return json_objects[-1] if json_objects else {}


def utils_get_json_blocks_from_text(text: str) -> list[dict]:
    json_blocks = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    json_objects = []
    for block in json_blocks:
        try:
            json_objects.append(json.loads(block))
        except json.JSONDecodeError:
            continue
    return json_objects if json_objects else []


def utils_get_python_blocks_from_text(text: str) -> list[str]:
    """
    Extracts the last Python code block from a given text string.

    Args:
        text (str): The input text containing Python code blocks.

    Returns:
        str: The last Python code block found in the text.
    """
    python_blocks = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    return python_blocks if python_blocks else ""


def is_openai_format(input_) -> bool:
    """Checks if the input is in OpenAI chat-like format:

    ```python
    [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
    ]
    ```

    Args:
        input_: The input to check.

    Returns:
        A boolean indicating if the input is in OpenAI chat-like format.
    """
    if not isinstance(input_, list):
        return False
    return all(
        isinstance(x, dict) and "role" in x.keys() and "content" in x.keys()
        for x in input_
    )


def utils_run_qatch(
        db_path: str, selected_col: str, tbl_name: str
) -> list[dict[Literal['test_category', 'query', 'question'], str]]:
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

    sqlite_connector = SqliteConnector(
        relative_db_path=db_path,
        db_name='',
    )

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
        sqlite_connector, column_to_include=selected_col, tbl_names=[tbl_name]
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


def utils_levenshtein_name_in(list_values: list, name: str) -> str:
    """
    Get the name with the most overlapping characters from a list of names.

    Args:
        list_values (list): List of names to compare.
        name (str): Name to compare against.

    Returns:
        str: The name in list_values most similar to name.
    """
    name = name.lower().strip()
    max_ratio = -1
    most_overlapping_name = ""
    for val in list_values:
        ratio = Levenshtein.normalized_similarity(val.lower().strip(), name)
        if ratio > max_ratio:
            max_ratio = ratio
            most_overlapping_name = val

    return most_overlapping_name


def utils_execute_python_code(python_code) -> dict:
    local_namespace = {}
    try:
        # Dynamically execute the provided UDF code in a local namespace
        exec(python_code, globals(), local_namespace)
    except Exception:
        pass
    return local_namespace
