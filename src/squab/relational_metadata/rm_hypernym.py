from litellm import override
from squab.relational_metadata.rm_entity_component import AbstractRelationalMetadata


HYPERNYM_SYSTEM_PROMPT = (
    "Given a specific table schema and a designated set of columns, your task is to generate a suitable and appropriate label for that particular set. "
    'A set is defined as a collection of column names that share a semantic relationship, which may include examples such as "First Name" and '
    '"Last Name." The label you create should be a single term that effectively encompasses all the columns in that set, such as "Personal Information" or "Name." '
    "It is crucial that the label must be unique within its set and the table schema and also relevant to the overall table schema in "
    "question. As output, return a JSON enclosed in ```json ```. Instead, if there is no possible labeling solution, return an empty dictionary ``json {}```. "
    "\n## Output\n\n"
    "```json\n"
    "{\n"
    '  "label": "the label that represents the set of columns"\n'
    "}\n"
    "```\n"
)

HYPERNYM_FEW_SHOTS = []

HYPERNYM_TEMPLATE = """
## Table Schema
{{ tbl_schema }}
## Semantic related columns
{{ cols }}
""".rstrip()


class RMHypernym(AbstractRelationalMetadata):
    system_prompt: str = HYPERNYM_SYSTEM_PROMPT
    few_shots_messages: list = HYPERNYM_FEW_SHOTS
    template: str = HYPERNYM_TEMPLATE

    @override
    def update_line(
        self, line_to_update, relational_metadata, relational_metadata_cost, rm_metadata
    ):
        relational_metadata["columns"] = line_to_update["pattern_identification"]
        line_to_update["relational_metadata"] = relational_metadata
        line_to_update["relational_metadata_cost"] = relational_metadata_cost
        line_to_update["rm_metadata"] = rm_metadata
        return line_to_update
    
    @override
    def get_messageges_with_user_question(self, line) -> str:
        tbl_schema = line["table"]["db_schema"]
        return self._messages + [
            {
                "role": "user",
                "content": self._template.render(
                    tbl_schema=tbl_schema, cols=line["pattern_identification"]
                ),
            },
        ]
