from typing import Literal

import sqlalchemy
from sqlalchemy import text

from squab.graph_states import Line
from squab.nodes.test_generation.utils_decorator_llm_node import test_generation_based_templates
from squab.nodes.utils import GenerationSteps, utils_run_qatch


@test_generation_based_templates()
def create_templates_col_unans(
        line: Line,
        *args,
        **kwargs
) -> list[list[dict[Literal['test_category', 'query', 'question'], str]]]:
    col_to_use_for_generation = line[GenerationSteps.RM.value].pop('col_to_use_for_generation')
    list_queries_with_selected_col = utils_run_qatch(
        db_path=line['db_path'],
        selected_col=col_to_use_for_generation,
        tbl_name=line["tbl_name"]
    )
    templates = []
    for test_query in list_queries_with_selected_col:
        test_query['query'] = test_query['query'].replace(f'`{col_to_use_for_generation}`',
                                                          f"`{line[GenerationSteps.RM.value]['column_name']}`")
        if _check_unanswerability_query(test_query['query'], line['db_path']):
            test_query['question'] = test_query['question'].replace(col_to_use_for_generation,
                                                                    line[GenerationSteps.RM.value]['column_name'])
            templates.append([test_query])

    return templates


def _check_unanswerability_query(query: str, db_path: str) -> bool:
    try:
        engine = sqlalchemy.create_engine(f'sqlite:///{db_path}')
        result = engine.connect().execute(text(query))
    except sqlalchemy.exc.OperationalError as e:
        if 'no such column' in str(e).lower():
            return True
        return False
    else:
        return False
