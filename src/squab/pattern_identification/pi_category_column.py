import litellm
from typing import TYPE_CHECKING, override
from litellm.cost_calculator import completion_cost
from distilabel.steps import StepInput
from pydantic import Field

from squab.pattern_identification.abstract_pattern_identification import (
    AbstractPatternIdentification,
)
from squab.utils import utils_levenshtein_name_in
from squab.utils.utils_get_columns_no_pk_no_fk import utils_get_columns_no_pk_fk
from squab.utils.utils_get_last_json_from_text import utils_get_last_json_from_text

if TYPE_CHECKING:
    from distilabel.typing import StepOutput


class PICategoryColumn(AbstractPatternIdentification):
    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="Name of the generative model to use to fine the type column.",
    )

    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        count = 0
        dataset = []

        for line in inputs:
            table = line["table"]
            column_names = utils_get_columns_no_pk_fk(
                table, start_from_cols=list(table["cat_col2metadata"].keys())
            )
            pi_metadata: dict[str, str] = {"pattern_type": "type_column"}
            type_columns, cost = self._get_type_column_and_cost(
                columns=column_names,
                table=table,
            )

            for type_column in list(type_columns.values())[0]:
                # get the the column name with the most syntactic overlap

                type_column = utils_levenshtein_name_in(
                    list_values=column_names, name=type_column
                )

                line_updated = self.update_line(
                    line,
                    pattern_identification=type_column,
                    pattern_identification_cost=cost,
                    pi_metadata=pi_metadata,
                )
                dataset.append(line_updated)
                count += 1
                if count >= self.max_identified_patterns_per_tbl:
                    self._logger.info(
                        "Maximum number of patterns reached for this table."
                    )
                    break
        yield dataset

    def _get_type_column_and_cost(
        self, columns: list[str], table: dict
    ) -> tuple[list[str] | None, float]:
        messages = [
            {
                "role": "user",
                "content": (
                    f'Given the table "{table["tbl_name"]}" with the following columns and example values:\n\n'
                    f"{table['db_schema_table_examples']}\n\n"
                    "Identify the columns that represent categories or types defining the classification of the records. "
                    "These are typically columns with a limited set of repeated values, such as product categories, departments, or statuses. "
                    "Provide the answer in JSON format with the key 'type_columns' and a list of the identified column names as the value. "
                    "If no such columns exist, return an empty list.\n\n"
                    "For example, if the table is 'Cars' with columns and example values:\n"
                    '["CarID": 1, "Make": "Toyota", "Model": "Corolla", "Year": 2020],\n'
                    '["CarID": 2, "Make": "Honda", "Model": "Civic", "Year": 2021],\n'
                    '["CarID": 3, "Make": "Ford", "Model": "Focus", "Year": 2020]\n'
                    "The answer should be:\n"
                    '{"type_columns": ["Make", "Model"]}'
                ),
            }
        ]
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
        )
        cost = completion_cost(completion_response=response)
        selected_columns = utils_get_last_json_from_text(
            response["choices"][0]["message"]["content"]
        )
        return selected_columns, cost
