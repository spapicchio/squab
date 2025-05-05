from dataclasses import Field, field
from typing import TYPE_CHECKING, Any, List, Union
from typing_extensions import override

from distilabel.steps.tasks import TextGeneration

if TYPE_CHECKING:
    from distilabel.typing import ChatType, StepColumns

from squab.utils import utils_get_last_json_from_text

c = (
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
    "  \"entity\": \"the name that represents the entity\",\n"
    "  \"component\": \"the name that represents the component.\"\n"
    "}\n"
    "```\n"
)

SYSTEM_PROMPT = """
You are a helpful AI assistant. Identify the semantic relationship between two provided names and determine if one is an Entity and the other is a Component. Note that a component can also be an element present in the entities.

# Steps
1. Analyze the first name to determine if it can be categorized as an Entity or a Component.
2. Analyze the second name to determine if it can be categorized as a Component or an Entity.
3. Evaluate if the selected component is a meaningful part or attribute of the selected entity.

# Output Format
Return the answer as JSON enclosed in ```json ``` with two keys: entity and component.
```json
{
  "entity": "the name that represents the entity",
  "component": "the name that represents the component."
}
```
# Examples
**Example 1:**
- Input: "Engine", "Car"
- Output: 
```json
{
  "entity": "Car",
  "component": "Engine"
}
```
**Example 2:**
- Input: "Brand name", "Store name"
- Output:
```json
{
  "entity": "Store name",
  "component": "Brand name"
}
```
**Example 3:**
- Input: "Hospital", "Amenities"
- Output:
```json
{
  "entity": "Hospital",
  "component": "Amenities"
}
```
""".rstrip()


OUTPUT_NAME = "RMEntityComponent"
INPUT_NAME = "PIManyToMany"


class RMEntityComponent(TextGeneration):
    template: str = field(
        default="{{ PIManyToMany }}",
        description=("This is a template or prompt to use for the generation."),
    )
    system_prompt: str = Field(
        default=SYSTEM_PROMPT,
        description=("This is the system prompt to use for the generation. "),
    )

    columns: Union[str, List[str]] = Field(
        default=[INPUT_NAME],
        description=(
            "Custom column or list of columns to include in the input for the prompt. "
        ),
    )
    messages: List[dict] = Field(
        default_factory=list,
        description=(
            "List of messages to include in the input for the prompt. "
            "The messages should be in the format of a list of dictionaries with "
            "the keys 'role' and 'content'."
        ),
    )
    
    @override
    def model_post_init(self, __context: Any) -> None:
        self.columns = [self.columns] if isinstance(self.columns, str) else self.columns
        super().model_post_init(__context)
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Engine, Car"},
            {"role": "assistant", "content":(
                "```json\n"
                "{\n"
                "  \"entity\": \"Car\",\n"
                "  \"component\": \"Engine\"\n"
                "}\n"
                "```\n"
            )},
            {"role": "user", "content": "Brand name, Store name"},
            {"role": "assistant", "content": (
                "```json\n"
                "{\n"
                "  \"entity\": \"Store name\",\n"
                "  \"component\": \"Brand name\"\n"
                "}\n"
                "```\n"
            )},
            {"role": "user", "content": "Hospital, Amenities"},
            {"role": "assistant", "content": (
                "```json\n"
                "{\n"
                "  \"entity\": \"Hospital\",\n"
                "  \"component\": \"Amenities\"\n"
                "}\n"
                "```\n"
            )},
        ]



    def format_input(self, input: dict) -> "ChatType":
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        # Handle the previous expected errors, in case of custom columns there's more freedom
        # and we cannot check it so easily.
        user_message = self._prepare_message_content(input)
        return self.messages + user_message
    
    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return [OUTPUT_NAME, "model_name"]

    @override
    def format_output(self, output: str, input=None) -> dict:
        """The output is formatted as a dictionary with the `generation`. The `model_name`
        will be automatically included within the `process` method of `Task`."""
        # transform the string into a dictionary
        output_dict = utils_get_last_json_from_text(output)

        return {OUTPUT_NAME: output_dict}