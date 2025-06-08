import functools
from typing import Callable, TypeVar, Any

from squab.graph_states import Line
from squab.nodes.utils import GenerationSteps
from squab.nodes.utils_decorator_process_dataset import dataset_processor
from squab.nodes.utils_node_llm_call import llm_node_update_line

_T = TypeVar("_T")
_TCo = TypeVar("_TCo", covariant=True)


def test_generation_based_templates():
    """
    Decorator
    """

    def decorator(create_templates_from_line: Callable[[Line, Any], _T | list[_T]]):
        @functools.wraps(create_templates_from_line)
        @dataset_processor()
        def wrapper(line: Line,
                    tg_system: str | None,
                    tg_user: str,
                    tg_few_shots: list | None,
                    tg_litellm_params: dict,
                    user_template_params_from_line: dict[str, str],
                    *args,
                    **kwargs) -> list[list[dict]]:
            processed_lines = []

            for templates in create_templates_from_line(line, *args, **kwargs):
                sql_interpretations = [
                    template["query"] for template in templates
                ]
                line = llm_node_update_line(
                    line=line,
                    system=tg_system,
                    user=tg_user,
                    few_shots=tg_few_shots,
                    col_to_update='question',
                    litellm_params=tg_litellm_params,
                    template_params={key: line[val] for key, val in user_template_params_from_line.items()},
                    step=GenerationSteps.TG
                ).result()

                if 'question' not in line:
                    line['has_failed'] = {
                        GenerationSteps.TG.value: f"Model Response: {line['question']}"
                    }
                else:
                    line['question'] = line['question']['question']
                line['target'] = sql_interpretations
                line['test_sub_category'] = templates[0]['test_category']
                line['templates'] = templates
                processed_lines.append(line)

            if len(processed_lines) == 0:
                kwargs['logger'].warning(
                    f"No templates generated for line"
                )
                line['has_failed'] = {
                    GenerationSteps.TG.value: "No templates generated."
                }
                processed_lines.append(line)

            return processed_lines

        return wrapper

    return decorator
