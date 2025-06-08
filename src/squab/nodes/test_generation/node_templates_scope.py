from typing import Literal

from squab.graph_states import Line
from squab.nodes.test_generation.utils_decorator_llm_node import test_generation_based_templates
from squab.nodes.utils import GenerationSteps


@test_generation_based_templates()
def create_templates_scope(
        line: Line,
        *args,
        **kwargs
) -> list[list[dict[Literal['test_category', 'query', 'question'], str]]]:
    tbl_name = line["tbl_name"]
    entity = line[GenerationSteps.RM.value]["entity"]
    component = line[GenerationSteps.RM.value]["component"]

    query_1 = (
        f"SELECT `{component}` "
        f"FROM `{tbl_name}` "
        f"GROUP BY `{component}` "
        f"HAVING COUNT(DISTINCT `{entity}`) = (SELECT COUNT(DISTINCT `{entity}`) FROM `{tbl_name}`)"
    )
    question_1 = f"What {component.lower()} are present in all {entity.lower()}s?"

    query_2 = f"SELECT DISTINCT `{component}`, `{entity}` FROM `{tbl_name}` "
    question_2 = f"What {component.lower()} does each {entity.lower()} have?"
    question_sql_templates = [
        {
            "question": question_1,
            "query": query_1,
            "test_category": "",
        },
        {
            "question": question_2,
            "query": query_2,
            "test_category": "",
        },
    ]
    return [question_sql_templates]
