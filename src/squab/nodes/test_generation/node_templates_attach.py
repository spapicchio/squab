from squab.graph_states import Line
from squab.nodes.test_generation.utils_decorator_llm_node import test_generation_based_templates
from squab.nodes.utils import GenerationSteps


@test_generation_based_templates()
def create_templates_attach(
        line: Line,
        *args,
        **kwargs
) -> list[list[dict[str, str]]]:
    """
    Create templates for attachment category.
    """
    tbl_name = line["tbl_name"]
    entity = line[GenerationSteps.PI.value]["entity"]
    component = line[GenerationSteps.PI.value]["component"]
    column_project = line[GenerationSteps.PI.value]["column_to_project"]
    value_group_1, value_group_2 = line[GenerationSteps.PI.value]["entity_values"]
    value_intersection = line[GenerationSteps.PI.value]["component_value"]

    condition_1 = (
        f"(`{entity}` = '{value_group_1}' OR `{entity}` = '{value_group_2}')"
        f" AND `{component}` = '{value_intersection}'"
    )

    condition_2 = (
        f"`{entity}` = '{value_group_1}' "
        f"OR `{entity}` = '{value_group_2}' AND `{component}` = '{value_intersection}'"
    )

    question_sql_templates = [
        {
            "question": f"List {column_project} where {component} is {value_intersection} and {entity} is either {value_group_1} or {value_group_2}",
            "query": f"SELECT `{column_project}` FROM `{tbl_name}` WHERE {condition_1}",
            "test_category": "",
        },
        {
            "question": f"List {column_project} where {entity} is {value_group_1} or where {entity} is {value_group_2} and {component} is {value_intersection}",
            "query": f"SELECT `{column_project}` FROM `{tbl_name}` WHERE {condition_2}",
            "test_category": "",
        },
    ]
    return [question_sql_templates]
