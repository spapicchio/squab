from typing import TypedDict, Any


class Line(TypedDict):
    task_type: str
    table: dict | None
    db_id: str | None
    db_path: str | None
    db_schema: str | None
    db_schema_table: str | None
    db_schema_table_examples: str | None
    tbl_name: str | None
    pattern_identification: Any
    relational_metadata: Any
    question: str | None
    target: Any
    total_cost: float
