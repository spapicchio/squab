import copy
import random

from squab.graph_states import Line
from squab.nodes.utils import GenerationSteps
from squab.nodes.utils_decorator_process_dataset import dataset_processor
from squab.nodes.utils_node_llm_call import llm_node_update_line


@dataset_processor()
def create_new_attributes_from(
        line: Line,
        litellm_params_new_attributes: dict,
        new_attributes_system_template: str | None = None,
        new_attributes_user_template: str | None = None,
        *args, **kwargs
) -> list[Line]:
    """Process a single line for hypernym generation."""
    random.seed(kwargs.get('seed', 42))
    processed_lines = []

    line = llm_node_update_line(
        line,
        system=new_attributes_system_template,
        user=new_attributes_user_template,
        few_shots=[],
        col_to_update=GenerationSteps.RM.value,
        litellm_params=litellm_params_new_attributes,
        template_params={'tbl_schema': line['db_schema_table_examples']},
        step=GenerationSteps.RM,
    ).result()

    if 'suggested_columns' not in line[GenerationSteps.RM.value]:
        line['has_failed'] = {
            GenerationSteps.RM.value: f"Model Response: {line[GenerationSteps.RM.value]}"
        }
        processed_lines.append(line)

    else:
        suggested_columns = copy.deepcopy(line[GenerationSteps.RM.value]['suggested_columns'])
        total_cost = line['granular_costs'][GenerationSteps.RM.value]
        for new_col in suggested_columns:
            if "column_name" not in new_col or "column_type" not in new_col:
                continue
            new_line = copy.deepcopy(line)
            new_line['total_cost'] = new_line['total_cost'] - total_cost + total_cost / len(suggested_columns)
            new_line['granular_costs'][GenerationSteps.RM.value] = total_cost / len(suggested_columns)

            cat_col = random.choice(list(line['cat_col2metadata'].keys())) if line['cat_col2metadata'] else None
            num_col = random.choice(list(line['num_col2metadata'].keys())) if line['num_col2metadata'] else None
            selected_col = cat_col if new_col['column_type'] == 'categorical' else num_col

            new_line[GenerationSteps.RM.value] = {**new_col, 'col_to_use_for_generation': selected_col}
            processed_lines.append(new_line)

    return processed_lines
