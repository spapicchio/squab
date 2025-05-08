from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from typing_extensions import override
from pydantic import Field
from distilabel.steps import Step, StepInput

if TYPE_CHECKING:
    from distilabel.typing import StepColumns, StepOutput


class AbstractPatternIdentification(Step, ABC):
    max_identified_patterns_per_tbl: int = Field(
        default=10, description="Maximum number of patterns to identify per table."
    )

    @property
    def inputs(self) -> "StepColumns":
        return ["table"]

    @property
    def outputs(self) -> "StepColumns":
        return ["pattern_identification", "pattern_identification_cost", "pi_metadata"]

    @abstractmethod
    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        raise NotImplementedError(
            "The process method must be implemented in the subclass."
        )

    def update_line(
        self, line_to_update, pattern_identification, pattern_identification_cost, pi_metadata
    ):
        line_to_update["pattern_identification"] = pattern_identification
        line_to_update["pattern_identification_cost"] = pattern_identification_cost
        line_to_update["pi_metadata"] = pi_metadata
        return line_to_update
