import re

from langgraph.func import task
from qatch.connectors import SqliteConnector, ConnectorTable

from src.squab.graph_states import Line


@task
def node_read_db_sqlite(
        db_path: str,
        db_id: str | None = None,
        only_these_tbl: str | list | None = None,
        *args,
        **kwargs
) -> list[Line]:
    db_id = db_id or db_path.split("/")[-1].split(".")[0]
    tables = _load_qatch(db_path, db_id)
    if only_these_tbl is not None:
        tables = [tbl for tbl in tables if tbl['tbl_name'] in only_these_tbl]
    return tables


def _load_qatch(db_path: str, db_id: str) -> list[Line]:
    db_uri = f"file:{db_path}?immutable=1&uri=true"
    connector = SqliteConnector(
        relative_db_path=db_uri,
        db_name=db_id,
    )
    tbl_name2table: dict[str, ConnectorTable] = (
        connector.load_tables_from_database()
    )

    tables = [val.model_dump() for val in tbl_name2table.values()]
    for val in tables:
        val['db_path'] = db_path
        val["db_id"] = val.pop("db_name", None)
        val["db_schema_table"] = connector.run_query(
            f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{val['tbl_name']}'"
        )[0][0]
        val["db_schema_table_examples"] = (
            _create_database_schema_with_examples(val, val["db_schema_table"])
        )

        val["db_schema"] = "\n".join(
            [val[0] for val in connector.run_query("SELECT sql FROM sqlite_master")]
        )

        for col_name in val["foreign_keys"]:
            col_name["parent_column"] = str(col_name["parent_column"])

    return tables


def _create_database_schema_with_examples(table: dict, tbl_schema: str) -> str:
    """
    Insert example values as comments for each column in the dumped schema.
    """
    col_metadata = table["tbl_col2metadata"]
    lines = []
    for line in tbl_schema.split(","):
        stripped = line.strip()
        # Try to match column definitions (skip lines that start with CREATE TABLE, constraints, etc.)
        if (
                stripped
                and not stripped.upper().startswith("PRIMARY KEY")
                and not stripped.upper().startswith("FOREIGN KEY")
        ):
            columns_in_line = re.findall(
                r"[`'\"]?(\w+)[`'\"]?\s+\w+", stripped, re.IGNORECASE
            )
            for col_name in columns_in_line:
                meta = col_metadata.get(col_name, None)
                if meta:
                    sample_data = meta.get("sample_data", [])
                    sample_str = ", ".join(
                        [f"`{str(val)}`" for val in sample_data[:2]]
                    )
                    if sample_str:
                        line = line.rstrip() + f" -- Example Values: ({sample_str})"
        lines.append(line)
    return ",\n".join(lines)
