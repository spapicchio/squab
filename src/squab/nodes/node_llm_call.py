import litellm
from langgraph.func import task
from litellm.types.utils import ModelResponse

from squab.nodes.utils import is_openai_format


@task
def llm_call(messages, litellm_params) -> tuple[ModelResponse, float]:
    if not is_openai_format:
        raise ValueError(f"Only OpenAI format is supported. Received messages: {messages}")
    response = litellm.completion(
        messages=messages,
        **litellm_params
    )
    cost = litellm.completion_cost(response)
    return response, cost
