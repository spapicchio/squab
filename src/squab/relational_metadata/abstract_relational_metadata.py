from abc import ABC, abstractmethod
from os import system

from typing_extensions import TYPE_CHECKING, override


from jinja2 import Template
from pydantic import Field, PrivateAttr
from distilabel.steps import Step, StepInput

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

    system_prompt: str = Field(
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
        self._messages = [{"role": "system", "content": self.system_prompt}]

    @property
    def outputs(self) -> "StepColumns":
        return ["relational_metadata", "relational_metadata_cost", "rm_metadata"]

    @override
    def load(self) -> None:
        super().load()
        self._template = Template(self.template)
        self._messages = self._messages + self.few_shots_messages

    @abstractmethod
    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        raise NotImplementedError(
            "The process method must be implemented in the subclass."
        )

    def update_line(
        self, line_to_update, relational_metadata, relational_metadata_cost, rm_metadata
    ):
        line_to_update["relational_metadata"] = relational_metadata
        line_to_update["relational_metadata_cost"] = relational_metadata_cost
        line_to_update["rm_metadata"] = rm_metadata
        return line_to_update

    def get_messageges_with_user_question(self, **kwargs) -> str:
        return self._messages + [
            {"role": "user", "content": self._template.render(**kwargs)}
        ]
