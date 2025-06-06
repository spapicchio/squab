from enum import Enum

from src.squab.nodes.read_sqlite_db import read_db_sqlite


class Generators(Enum):
    SCOPE = "scope"
    ATTACHMENT = "attachment"
    VAGUE = "vague"
    COL_UNANS = "col_unans"
    CALC_UNANS = "calc_unans"
    OUT_OF_SCOPE = "out_of_scope"


def get_generators_callable(generators_name: str | list[str]):
    if isinstance(generators_name, str):
        generators_name = [generators_name]

    return ...


__ALL__ = [
    'read_db_sqlite'
]
