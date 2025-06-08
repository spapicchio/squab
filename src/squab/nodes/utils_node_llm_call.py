import copy

import litellm
from jinja2 import Template
from langgraph.func import task
from litellm.types.utils import ModelResponse

from squab.graph_states import Line
from squab.nodes.utils import is_openai_format, utils_get_last_json_from_text, GenerationSteps


def _create_messages(system: str | None, user: str, few_shots: list[dict], **kwargs):
    few_shots_messages = []
    for shot in few_shots:
        few_shots_messages.extend([
            {"role": "user", "content": Template(user).render(**shot)},
            {"role": "assistant", "content": shot['assistant_answer']}
        ])
    system_message = [{"role": "system", "content": system}] if system is not None else []
    user_message = [{"role": "user", "content": Template(user).render(**kwargs)}]
    return system_message + few_shots_messages + user_message


@task
def llm_call(messages: list, litellm_params: dict) -> tuple[ModelResponse, float]:
    if not is_openai_format:
        raise ValueError(f"Only OpenAI format is supported. Received messages: {messages}")
    response = litellm.completion(
        messages=messages,
        **litellm_params
    )
    cost = litellm.completion_cost(response)
    return response, cost


@task
def llm_node_update_line(line: Line,
                         system: str | None,
                         user: str,
                         few_shots: list[dict],
                         col_to_update: str,
                         litellm_params: dict,
                         template_params: dict,
                         step: GenerationSteps) -> Line:
    line = copy.deepcopy(line)
    messages = _create_messages(system, user, few_shots, **template_params)
    response, total_cost = llm_call(messages, litellm_params).result()
    model_response = response["choices"][0]["message"]["content"]
    generated_response = utils_get_last_json_from_text(model_response)
    if not generated_response:
        line[col_to_update] = ''
        line['has_failed'] = {
            step.value: f"Model Response: {model_response}"
        }
    else:
        line[col_to_update] = generated_response

    line['total_cost'] += total_cost
    line['granular_costs'][step.value] = total_cost
    return line
