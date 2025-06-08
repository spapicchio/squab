import copy

import pandas as pd
from sqlalchemy import create_engine

from squab.graph_states import Line
from squab.nodes.utils import GenerationSteps, utils_get_columns_no_pk_fk
from squab.nodes.utils_decorator_process_dataset import dataset_processor


@dataset_processor()
def get_many_to_many_from_line(
        line: Line,
        *args,
        **kwargs
):
    engine = create_engine(f"sqlite:///{line['db_path']}")
    columns_no_pk_fk = utils_get_columns_no_pk_fk(line)
    line['total_cost'] += 0.0
    line['granular_costs']['pattern_identification'] = 0.0
    identified_patterns = []
    for i in range(len(columns_no_pk_fk) - 1):
        for j in range(i + 1, len(columns_no_pk_fk)):
            col1, col2 = columns_no_pk_fk[i], columns_no_pk_fk[j]
            if _is_many_to_many(
                    line["tbl_name"], col1=col1, col2=col2, engine=engine
            ):
                identified_patterns.append([col1, col2])
    lines = []
    for pattern in identified_patterns:
        line[GenerationSteps.PI.value] = pattern
        lines.append(copy.deepcopy(line))

    if len(lines) == 0:
        line['has_failed'] = {
            GenerationSteps.PI.value: "No many-to-many relationships found."
        }
        lines.append(line)

    return lines


def _is_many_to_many(tbl_name: dict, col1: str, col2: str, engine) -> bool:
    # check for multiple unique values in col2 for each unique value in col1
    query = f"SELECT `{col1}`, `{col2}` FROM `{tbl_name}` LIMIT 100;"

    df = pd.read_sql_query(query, engine.connect())

    # check for multiple unique values in col1 for each unique value in col2
    cond1 = any(df.groupby(col1)[col2].nunique() > 1)
    cond2 = any(df.groupby(col2)[col1].nunique() > 1)

    return cond1 and cond2
