import sqlite3
from typing import Literal

from squab.graph_states import Line
from squab.nodes.test_generation.utils_decorator_llm_node import test_generation_based_templates
from squab.nodes.utils import GenerationSteps, utils_run_qatch, utils_execute_python_code


@test_generation_based_templates()
def create_templates_udf_unans(
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
                                                          f"{line[GenerationSteps.RM.value]['udf_name']}")
        if _execute_udf_query(
                test_query['query'],
                udf_python_code=line[GenerationSteps.RM.value]['udf_python_code'],
                udf_name=line[GenerationSteps.RM.value]['udf_name'],
                db_path=line['db_path']):
            test_query['question'] = test_query['question'].replace(col_to_use_for_generation,
                                                                    line[GenerationSteps.RM.value]['udf_name'])
            templates.append([test_query])

    return templates


def _execute_udf_query(query: str, udf_python_code: str, db_path: str, udf_name: str) -> bool:
    local_namespace = utils_execute_python_code(udf_python_code)
    func_name, func = local_namespace.popitem()
    num_arguments = len(udf_name.split('(')[1].split(','))

    try:
        # get a raw connection and create the UDF
        with sqlite3.connect(db_path) as conn:
            conn.create_function(func_name, num_arguments, func)
            # Execute the query to test the function
            cursor = conn.execute(query)
            cursor.fetchall()
    except sqlite3.OperationalError as e:
        if 'user-defined function' in str(e).lower():
            # If the error is due to the UDF pytcho code we consider it a success because the query run.
            return True
        return False
    return True
