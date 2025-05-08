from abc import ABC, abstractmethod

from typing_extensions import TYPE_CHECKING, override

from jinja2 import Template
from pydantic import Field, PrivateAttr
from distilabel.steps import Step, StepInput

from squab.utils.utils_is_open_ai_format import is_openai_format

if TYPE_CHECKING:
    from distilabel.typing import StepColumns, StepOutput


DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant who writes a natural language (NL) question. 
You are provided with a definition of ambiguity, the SQL queries that answer the question following the ambiguity rules, and a database containing the answers. You may also receive metadata helping you in generating the question. Your task is to write the NL question following these guidelines:

- All unformatted table and column names must be replaced with plain words, preferably synonyms.
- Make the question as short as possible, but do not miss any part of the question like order-by (e.g., remove unnecessary words or paraphrase). Yet, you must check the relevant tables to ensure that the question and its interpretations express the same request as the queries and would yield the same answer. Example: You can modify "fitness training program" into "training program" and omit the unnecessary word “fitness” only if "training program"  cannot be confused with other columns in different tables.
- You must maintain ambiguity when writing the question and reading each interpretation.
- If the projected column name can be inferred, remove it from the final output

# Output Format
Provide the answer in JSON format as follows
```json
{
    "question": "the generated question"
}
```
"""

DEFAULT_TEMPLATE = """
## Ambiguity Definition
{{ ambig_definition }}

## queries
{{ queries }}

## Metadata
{{ metadata }}

## Database
{{ database }}
""".rstrip()


class AbstractTestGeneration(Step, ABC):
    template: str = Field(
        default=DEFAULT_TEMPLATE,
    )

    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt for the model.",
    )

    ambiguity_definition: str

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
        return ["table", "pattern_identification", "relational_metadata"]

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
        return ["test_question", "test_target", "test_cost", "test_metadata"]

    @abstractmethod
    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        raise NotImplementedError(
            "The process method must be implemented in the subclass."
        )

    def update_line(
        self, line_to_update, test_question, test_target, test_cost, test_metadata
    ):
        line_to_update["test_question"] = test_question
        line_to_update["test_target"] = test_target
        line_to_update["test_cost"] = test_cost
        line_to_update["test_metadata"] = test_metadata
        return line_to_update

    @override
    def load(self) -> None:
        super().load()
        self._template = Template(self.template)

        self._messages = self._messages + self.few_shots_messages

    def get_messageges_with_user_question(self, **kwargs) -> str:
        return self._messages + [
            {"role": "user", "content": self._template.render(**kwargs)}
        ]
