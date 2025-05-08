from typing import Any, Dict, Union, TYPE_CHECKING

from distilabel.steps.tasks import Task

if TYPE_CHECKING:
    from distilabel.typing import StepColumns, ChatType


class PISemanticallyCloseColumns(Task):
    @property
    def inputs(self) -> "StepColumns":
        return ["input_field"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        return [
            {
                "role": "user",
                "content": input["input_field"],
            },
        ]

    @property
    def outputs(self) -> "StepColumns":
        return ["output_field", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"output_field": output}