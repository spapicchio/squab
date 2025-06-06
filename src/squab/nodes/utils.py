import json
import re

from squab.graph_states import Line
from squab.nodes.generation_steps import GenerationSteps


def utils_check_previous_step(dataset: list[Line], step: GenerationSteps) -> None:
    """
    Check if the previous step has been executed by looking for a specific key in the dataset.
    """
    if step == GenerationSteps.PI:
        keys = ["db_id", "db_path", "db_schema", "db_schema_table", "db_schema_table_examples", "tbl_name"]
        previous_step = 'reading_table'
    elif step == GenerationSteps.RM:
        keys = ["pattern_identification"]
        previous_step = GenerationSteps.PI.value
    elif step == GenerationSteps.TG:
        keys = ["relational_metadata"]
        previous_step = GenerationSteps.RM.value
    else:
        raise ValueError(f"Unknown step: {step}")

    for line in dataset:
        if not all(key in line for key in keys):
            raise ValueError(f"Previous step {previous_step} has not been executed. Missing keys: {keys}")


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
    json_objects = re.findall(r'\{.*?\}', text, re.DOTALL)

    if not json_objects:
        return {}

    # Parse the last JSON object
    last_json_str = json_objects[-1]
    try:
        return json.loads(last_json_str)
    except json.JSONDecodeError:
        return {}


def is_openai_format(input) -> bool:
    """Checks if the input is in OpenAI chat-like format:

    ```python
    [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
    ]
    ```

    Args:
        input: The input to check.

    Returns:
        A boolean indicating if the input is in OpenAI chat-like format.
    """
    if not isinstance(input, list):
        return False
    return all(
        isinstance(x, dict) and "role" in x.keys() and "content" in x.keys()
        for x in input
    )
