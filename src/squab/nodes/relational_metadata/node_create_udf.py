import copy
import random

from squab.graph_states import Line
from squab.nodes.utils import (
    GenerationSteps,
    utils_execute_python_code,
    utils_get_json_blocks_from_text, utils_get_python_blocks_from_text
)
from squab.nodes.utils_decorator_process_dataset import dataset_processor
from squab.nodes.utils_node_llm_call import _create_messages, llm_call


@dataset_processor()
def create_udf_from(
        line: Line,
        litellm_params_udf: dict,
        udf_system_template: str | None = None,
        udf_user_template: str | None = None,
        udf_generation_examples: list[dict] = (),
        *args, **kwargs
) -> list[Line]:
    random.seed(kwargs.get('seed', 42))
    messages = _create_messages(
        udf_system_template,
        udf_user_template,
        few_shots=udf_generation_examples,
        tbl_schema=line['db_schema_table_examples']
    )
    response, total_cost = llm_call(messages, litellm_params_udf).result()
    model_response = response["choices"][0]["message"]["content"]
    model_responses = extract_udf_python_from_model_response(model_response)

    processed_lines = []
    total_cost = total_cost / len(model_response)
    for udf, udf_python_code in model_responses:
        local_namespace = utils_execute_python_code(udf_python_code)
        if (
                len(udf) == 0 or
                len(udf_python_code) == 0 or
                'udf_name' not in udf or
                'udf_output_type' not in udf or
                len(local_namespace) == 0
        ):
            line['has_failed'] = {
                'create_unans_udf_from': f"Model Response: {model_response}"
            }
            processed_lines.append(line)
            break

        new_line = copy.deepcopy(line)
        new_line['total_cost'] += total_cost
        new_line['granular_costs'][GenerationSteps.RM.value] = total_cost

        cat_col = random.choice(list(line['cat_col2metadata'].keys())) if line['cat_col2metadata'] else None
        num_col = random.choice(list(line['num_col2metadata'].keys())) if line['num_col2metadata'] else None
        selected_col = cat_col if udf['udf_output_type'] == 'categorical' else num_col

        new_line[GenerationSteps.RM.value] = {**udf,
                                              'col_to_use_for_generation': selected_col,
                                              'udf_python_code': udf_python_code}
        processed_lines.append(new_line)

    return processed_lines


def extract_udf_python_from_model_response(model_response: str) -> list[tuple[dict, str]]:
    """
    Extracts the UDF Python code from the model response.
    The UDF Python code is expected to be in the format:
    ```python
    def udf_name(args):
        # function body
    ```
    """
    json_blocks = utils_get_json_blocks_from_text(model_response)
    python_blocks = utils_get_python_blocks_from_text(model_response)
    python_blocks = python_blocks \
        if len(python_blocks) > 0 else ['def placeholder():\n\tprint("hello")'] * len(json_blocks)
    return list(zip(json_blocks, python_blocks))
