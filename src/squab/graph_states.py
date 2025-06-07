from typing import Any

from qatch.connectors import ConnectorTable


class Line(ConnectorTable):
    task_type: str
    has_failed: dict | None
    table: dict | None  # PI
    db_id: str | None  # PI
    db_path: str | None  # PI
    db_schema: str | None  # PI
    db_schema_table: str | None  # PI
    db_schema_table_examples: str | None  # PI
    tbl_name: str | None  # PI
    pattern_identification: Any  # PI
    relational_metadata: Any  # RM
    question: str | None
    templates: list[dict] | None
    test_sub_category: str | None   # extracted from QATCH
    target: Any
    total_cost: float
    granular_costs: dict
