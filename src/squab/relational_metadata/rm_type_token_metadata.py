from typing import override
from pydantic import Field

from squab.relational_metadata.rm_entity_component import AbstractRelationalMetadata


TEMPLATE = (
    'Given the table "{{ tbl_name }}" with the following columns and example values:\n\n'
    "{{ formatted_columns_with_examples }}\n\n"
    "The column {{ pattern_identification }} has been identified as exhibiting type/token ambiguity, meaning its values can refer both to a general category (type) and to specific instances (tokens). "
    "Identify a term that encapsulates this duality—referring both to the entire table as a general category and to individual records as specific instances. "
    "Provide the answer in JSON format with the key 'ambiguous_term' and the identified term as the value. "
    "For example, if the table is 'Cars' with columns and example values:\n"
    '["CarID": 1, "Make": "Toyota", "Model": "Corolla", "Year": 2020],\n'
    '["CarID": 2, "Make": "Honda", "Model": "Civic", "Year": 2021],\n'
    '["CarID": 3, "Make": "Ford", "Model": "Focus", "Year": 2020]\n'
    "The answer should be:\n"
    '{"ambiguous_term": "car"}'
)


class RMCategoryTokenName(AbstractRelationalMetadata):
    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="Name of the generative model to use to fine the type column.",
    )
    template: str = Field(
        default=TEMPLATE,
        description="Template for the model to generate relational metadata.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="System prompt for the model.",
    )

    @override
    def get_messageges_with_user_question(self, line) -> str:
        return self._messages + [
            {
                "role": "user",
                "content": self._template.render(
                    tbl_name=line["table"]["tbl_name"],
                    formatted_columns_with_examples=line["table"][
                        "db_schema_table_examples"
                    ],
                    pattern_identification=line["pattern_identification"],
                ),
            }
        ]
