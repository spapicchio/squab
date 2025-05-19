from typing_extensions import TYPE_CHECKING, override


from distilabel.steps import StepInput

from squab.relational_metadata.abstract_relational_metadata import (
    AbstractRelationalMetadata,
)

if TYPE_CHECKING:
    from distilabel.typing import StepOutput


class RMOverlappingColValues(AbstractRelationalMetadata):
    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        # the metadata is extracted with an Heuristics during pattern identification for Attachment
        dataset = []
        for line in inputs:
            if line["pattern_identification"] is None:
                updated_line = self.update_line(
                    line,
                    None,
                    0.0,
                    None,
                )
                dataset.append(updated_line)
                continue

            relational_metadata = line["pattern_identification"]
            updated_line = self.update_line(
                line,
                relational_metadata,
                0.0,
                relational_metadata,
            )
            dataset.append(updated_line)
        yield dataset if dataset else [None]
