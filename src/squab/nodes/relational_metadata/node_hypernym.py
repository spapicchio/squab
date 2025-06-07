from squab.graph_states import Line
from squab.nodes.utils import GenerationSteps
from squab.nodes.utils_decorator_process_dataset import dataset_processor
from squab.nodes.utils_node_llm_call import llm_node_update_line


@dataset_processor()
def process_hypernym_line(
        line: Line,
        litellm_params_hypernym: dict,
        hypernym_system_template: str | None = None,
        hypernym_user_template: str | None = None,
        *args, **kwargs
) -> Line:
    """Process a single line for hypernym generation."""
    line = llm_node_update_line(
        line,
        system=hypernym_system_template,
        user=hypernym_user_template,
        few_shots=[],
        col_to_update=GenerationSteps.RM.value,
        litellm_params=litellm_params_hypernym,
        template_params={'tbl_schema': line['db_schema_table_examples'], 'cols': line[GenerationSteps.PI.value]},
        step=GenerationSteps.RM,
    ).result()
    if 'label' not in line[GenerationSteps.RM.value]:
        line['has_failed'] = {
            GenerationSteps.RM.value: f"Model Response: {line[GenerationSteps.RM.value]}"
        }

    return line
