from abc import ABC

import litellm
from litellm.cost_calculator import completion_cost
from typing_extensions import TYPE_CHECKING, override


from jinja2 import Template
from pydantic import Field, PrivateAttr
from distilabel.steps import Step, StepInput

from squab.utils.utils_get_last_json_from_text import utils_get_last_json_from_text
from squab.utils.utils_is_open_ai_format import is_openai_format

if TYPE_CHECKING:
    from distilabel.typing import StepColumns, StepOutput

DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant that identifies relational metadata.
"""

DEFAULT_TEMPLATE = """
{{ pattern_identification }}
"""


class AbstractRelationalMetadata(Step, ABC):
    template: str = Field(
        default=DEFAULT_TEMPLATE,
        description="Template for the model to generate relational metadata.",
    )

    system_prompt: str | None = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt for the model.",
    )

    few_shots_messages: list = Field(
        default_factory=list,
        description="List of few-shot examples to provide context for the model in OpenAI format.",
    )

    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="The name of the model to use for text generation.",
    )

    _template: Template | None = PrivateAttr(default=...)
    _messages: list[dict[str, str]] = PrivateAttr(default=...)

    @property
    def inputs(self) -> "StepColumns":
        return ["table", "pattern_identification"]

    @override
    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if not is_openai_format(self.few_shots_messages):
            raise ValueError(
                "Few shots messages must be in OpenAI format. Please check the format."
            )
        self._messages = (
            [{"role": "system", "content": self.system_prompt}]
            if self.system_prompt
            else []
        )

    @property
    def outputs(self) -> "StepColumns":
        return ["relational_metadata", "relational_metadata_cost", "rm_metadata"]

    @override
    def load(self) -> None:
        super().load()
        self._template = Template(self.template)
        self._messages = self._messages + self.few_shots_messages

    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        dataset = []
        for line in inputs:
            if not self.is_previous_step_correct(line):
                dataset.append(self.create_none_line(line))
                continue

            messages = self.get_messageges_with_user_question(line)

            response = litellm.completion(
                model=self.model_name,
                messages=messages,
            )

            rm_metadata = response.model_dump()
            rm_metadata["messages"] = messages
            relational_metadata_cost = completion_cost(completion_response=response)
            relational_metadata = utils_get_last_json_from_text(
                response["choices"][0]["message"]["content"]
            )

            if len(relational_metadata) > 0:
                updated_line = self.update_line(
                    line,
                    relational_metadata,
                    relational_metadata_cost,
                    rm_metadata,
                )
                dataset.append(updated_line)

        yield dataset

    def is_previous_step_correct(self, line) -> bool:
        return line["pattern_identification"] is not None

    def create_none_line(self, line):
        return self.update_line(
            line,
            None,
            0.0,
            None,
        )

    def update_line(
        self, line_to_update, relational_metadata, relational_metadata_cost, rm_metadata
    ):
        line_to_update["relational_metadata"] = relational_metadata
        line_to_update["relational_metadata_cost"] = relational_metadata_cost
        line_to_update["rm_metadata"] = rm_metadata
        return line_to_update

    def get_messageges_with_user_question(self, line) -> str:
        return self._messages + [
            {"role": "user", "content": self._template.render(**line)}
        ]
