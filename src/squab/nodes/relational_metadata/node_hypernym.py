from jinja2 import Template
from langgraph.func import task

from squab.graph_states import Line
from squab.nodes.node_llm_call import llm_call
from squab.nodes.node_process_dataset import node_process_dataset
from squab.nodes.utils import utils_get_last_json_from_text, GenerationSteps


@task
def node_get_hypernym(
        dataset: list[Line],
        litellm_params_hypernym: dict,
        hypernym_user_template: str | None = None,
        hypernym_system_template: str | None = None,
        *args, **kwargs
) -> list[Line]:
    """Generate hypernyms for the dataset using the provided templates and LLM parameters."""
    return node_process_dataset(
        dataset=dataset,
        step=GenerationSteps.RM,
        process_line_fn=process_hypernym_line,
        litellm_params_hypernym=litellm_params_hypernym,
        hypernym_user_template=hypernym_user_template,
        hypernym_system_template=hypernym_system_template,
        *args, **kwargs
    ).result()


HYPERNYM_SYSTEM_PROMPT = (
    "Given a specific table schema and a designated set of columns, your task is to generate a suitable and appropriate label for that particular set. "
    'A set is defined as a collection of column names that share a semantic relationship, which may include examples such as "First Name" and '
    '"Last Name." The label you create should be a single term that effectively encompasses all the columns in that set, such as "Personal Information" or "Name." '
    "It is crucial that the label must be unique within its set and the table schema and also relevant to the overall table schema in "
    "question. As output, return a JSON enclosed in ```json ```. Instead, if there is no possible labeling solution, return an empty dictionary ``json {}```. "
    "\n## Output\n\n"
    "```json\n"
    "{\n"
    '  "label": "the label that represents the set of columns"\n'
    "}\n"
    "```\n"
)

HYPERNYM_TEMPLATE = """
## Table Schema
{{ tbl_schema }}
## Semantic related columns
{{ cols }}
""".rstrip()


def process_hypernym_line(
        line: Line,
        litellm_params_hypernym: dict,
        hypernym_user_template: str | None = None,
        hypernym_system_template: str | None = None,
        *args, **kwargs
) -> Line:
    """Process a single line for hypernym generation."""
    system_prompt = hypernym_system_template or HYPERNYM_SYSTEM_PROMPT
    user_template = hypernym_user_template or HYPERNYM_TEMPLATE

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": Template(user_template).render(tbl_schema=line['db_schema_table_examples'],
                                                   cols=line['pattern_identification'])}
    ]

    response, total_cost = llm_call(messages, litellm_params_hypernym).result()
    model_response = response["choices"][0]["message"]["content"]
    hypernym = utils_get_last_json_from_text(model_response)

    if not hypernym:
        line['has_failed'] = {
            'rm_hypernym': f"The table has no hypernym, cannot find a pattern. Model Response: {model_response}"
        }
    else:
        line['relational_metadata'] = hypernym

    line['granular_costs']['relational_metadata'] = total_cost
    line['total_cost'] += total_cost

    return line
