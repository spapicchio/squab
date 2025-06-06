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
