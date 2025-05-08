def utils_get_columns_no_pk_fk(table: dict, start_from_cols: list[str] | None = None) -> list[str]:
    """
    Retrieves a list of column names excluding primary key, foreign key, and certain unwanted patterns.

    The function processes the provided or default column names of a table and excludes
    those that are primary keys, foreign keys, or contain specific substrings such as
    `id`, `code`, or `key` in their name.
    """
    column_names = set(start_from_cols or table['tbl_col2metadata'].keys())
    keys_to_remove = set(pk['column_name'] for pk in table['primary_key']) if table["primary_key"] else set()
    keys_to_remove.update(fk['parent_column'] for fk in table['foreign_keys']) if table["foreign_keys"] else None
    column_names -= keys_to_remove
    column_names = [val for val in column_names if
                    'id' not in val.lower() and
                    'code' not in val.lower() and
                    'key' not in val.lower()]
    return column_names