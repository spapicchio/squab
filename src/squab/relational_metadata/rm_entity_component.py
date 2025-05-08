from typing_extensions import TYPE_CHECKING
import litellm
from litellm.cost_calculator import completion_cost

from distilabel.steps import StepInput

from squab.relational_metadata.abstract_relational_metadata import (
    AbstractRelationalMetadata,
)
from squab.utils.utils_get_last_json_from_text import utils_get_last_json_from_text

if TYPE_CHECKING:
    from distilabel.typing import StepOutput


ENTITY_COMPONENT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Identify the semantic relationship between two provided names and determine "
    "if one is an Entity and the other is a Component. "
    "Note that a component can also be an element present in the entities."
    "\n\n# Steps\n"
    "1. Analyze the first name to determine if it can be categorized as an Entity or a Component.\n"
    "2. Analyze the second name to determine if it can be categorized as a Component or an Entity.\n"
    "3. Evaluate if the selected component is a meaningful part or attribute of the selected entity.\n\n"
    "# Output Format\n"
    "Return the answer as JSON enclosed in ```json ``` with two keys: entity and component.\n"
    "```json\n"
    "{\n"
    '  "entity": "the name that represents the entity",\n'
    '  "component": "the name that represents the component."\n'
    "}\n"
    "```\n"
)

ENTITY_COMPONENT_FEW_SHOTS = [
    {"role": "user", "content": "Engine, Car"},
    {
        "role": "assistant",
        "content": (
            '```json\n{\n  "entity": "Car",\n  "component": "Engine"\n}\n```\n'
        ),
    },
    {"role": "user", "content": "Brand name, Store name"},
    {
        "role": "assistant",
        "content": (
            "```json\n"
            "{\n"
            '  "entity": "Store name",\n'
            '  "component": "Brand name"\n'
            "}\n"
            "```\n"
        ),
    },
    {"role": "user", "content": "Hospital, Amenities"},
    {
        "role": "assistant",
        "content": (
            '```json\n{\n  "entity": "Hospital",\n  "component": "Amenities"\n}\n```\n'
        ),
    },
]


class RMEntityComponent(AbstractRelationalMetadata):
    system_prompt: str = ENTITY_COMPONENT_SYSTEM_PROMPT
    few_shots_messages: list = ENTITY_COMPONENT_FEW_SHOTS

    def process(self, inputs: StepInput) -> "StepOutput":
        dataset = []
        for line in inputs:
            messages = self._messages + [
                {"role": "user", "content": self._template.render(**line)}
            ]
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
