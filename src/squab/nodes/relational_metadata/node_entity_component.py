from squab.graph_states import Line
from squab.nodes.utils import GenerationSteps, utils_levenshtein_name_in
from squab.nodes.utils_decorator_process_dataset import dataset_processor
from squab.nodes.utils_node_llm_call import llm_node_update_line


@dataset_processor()
def find_entity_component_from(
        line: Line,
        litellm_params_entity_component: dict,
        entity_component_system_template: str,
        entity_component_user_template: str,
        entity_component_few_shots: list[dict],
        *args, **kwargs
) -> Line:
    """Process a single line for hypernym generation."""
    line = llm_node_update_line(
        line,
        system=entity_component_system_template,
        user=entity_component_user_template,
        few_shots=entity_component_few_shots,
        col_to_update=GenerationSteps.RM.value,
        litellm_params=litellm_params_entity_component,
        template_params={GenerationSteps.PI.value: line[GenerationSteps.PI.value],
                         'database': line["db_schema_table_examples"]},
        step=GenerationSteps.RM,
    ).result()

    if 'entity' not in line[GenerationSteps.RM.value] or 'component' not in line[GenerationSteps.RM.value]:
        line['has_failed'] = {
            GenerationSteps.RM.value: "Model Response: "
                                      f"{line[GenerationSteps.RM.value]}"
        }
    else:
        line[GenerationSteps.RM.value]['entity'] = utils_levenshtein_name_in(
            line['tbl_col2metadata'].keys(), line[GenerationSteps.RM.value]['entity']
        )
        line[GenerationSteps.RM.value]['component'] = utils_levenshtein_name_in(
            line['tbl_col2metadata'].keys(), line[GenerationSteps.RM.value]['component']
        )
    return line
