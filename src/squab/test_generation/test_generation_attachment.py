from typing_extensions import override, Literal

from squab.test_generation.abstract_test_generation import (
    DEFAULT_TEMPLATE,
    AbstractTestGeneration,
)


from jinja2 import Template


ATTACH_AMB_DEF = (
    "Attachment ambiguity: Attachment ambiguity refers to situations where two phrases "
    "are connected with relative pronouns, and it is ambiguous if the second phrase is "
    "attached to the end of the first phrase or the entire first phrase. "
    "The ambiguity rise when there is a many-to-many relationship between two columns that have a "
    "'Entity' - 'Component' semantic relation and distinct values in the Entity columns have same value in "
    "the Component column. In the question, it is ambiguous whether the value in the component column "
    "has to be attached to only one of the value in the Entity columns or to both. "
    "If possible, try to formulate the question without mentioning the column to project. "
)


ATTACH_FEW_SHOTS = [
    {
        "role": "user",
        "content": Template(DEFAULT_TEMPLATE).render(
            ambig_definition=ATTACH_AMB_DEF,
            queries="\n".join(
                [
                    'SELECT EventSpaces.Name \r\nFROM EventSpaces\r\nWHERE (EventSpaces.Event_Space = "Banquet Hall" OR EventSpaces.Event_Space = "Conference Room") AND EventSpaces.Capacity = 200',
                    'SELECT EventSpaces.Name \r\nFROM EventSpaces\r\nWHERE EventSpaces.Event_Space = "Banquet Hall" OR EventSpaces.Event_Space = "Conference Room" AND EventSpaces.Capacity = 200',
                ]
            ),
            metadata="""{"entity": "Event_Space", "component": "Capacity"}""",
            database="CREATE TABLE EventSpaces (\n   Name TEXT,\n    Event_Space TEXT,\n    Capacity INTEGER\n);",
        ),
    },
    {
        "role": "assistant",
        "content": """
        ```json
        {
            "question": ?List all banquet halls and conference rooms with a 200 person capacity."
        } 
        """,
    },
    {
        "role": "user",
        "content": Template(DEFAULT_TEMPLATE).render(
            ambig_definition=ATTACH_AMB_DEF,
            queries="\n".join(
                [
                    'SELECT MusicPerformer.Name \r\nFROM MusicPerformer\r\nWHERE (MusicPerformer.MusicPerformerType = "Jazz Musician" OR MusicPerformer.MusicPerformerType = "Rock Guitarist") AND MusicPerformer.YearsInIndustry = 10',
                    'SELECT MusicPerformer.Name \r\nFROM MusicPerformer\r\nWHERE MusicPerformer.MusicPerformerType = "Jazz Musician" OR MusicPerformer.MusicPerformerType = "Rock Guitarist" AND MusicPerformer.YearsInIndustry = 10',
                ]
            ),
            metadata="""{"entity": "MusicPerformerType", "component": "YearsInIndustry"}""",
            database="CREATE TABLE MusicPerformer (\n   Name TEXT,\n    MusicPerformerType TEXT,\n    YearsInIndustry INTEGER\n);",
        ),
    },
    {
        "role": "assistant",
        "content": """
        ```json
        {
            "question": "Display jazz musicians and rock guitarists who have been in the industry for 10 years."
        } 
        """,
    },
]


class TestGeneratorAttachment(AbstractTestGeneration):
    ambiguity_definition: str = ATTACH_AMB_DEF
    few_shots_messages: list = ATTACH_FEW_SHOTS

    @override
    def _create_question_sql_templates(
        self, line
    ) -> list[dict[Literal["query", "question", "test_category"]], str]:
        tbl_name = line["table"]["tbl_name"]
        entity = line["relational_metadata"]["entity"]
        component = line["relational_metadata"]["component"]
        column_project = line["relational_metadata"]["column_to_project"]

        value_group_1, value_group_2 = line["relational_metadata"]["entity_values"]
        value_intersection = line["relational_metadata"]["component_value"]

        condition_1 = (
            f"(`{entity}` = '{value_group_1}' OR `{entity}` = '{value_group_2}')"
            f" AND `{component}` = '{value_intersection}'"
        )
        condition_2 = (
            f"`{entity}` = '{value_group_1}' "
            f"OR `{entity}` = '{value_group_2}' AND `{component}` = '{value_intersection}'"
        )

        question_sql_templates = [
            {
                "question": "TODO",  # TODO
                "query": f"SELECT `{column_project}` FROM `{tbl_name}` WHERE {condition_1}",
                "test_category": "",
            },
            {
                "question": "TODO",  # TODO
                "query": f"SELECT `{column_project}` FROM `{tbl_name}` WHERE {condition_2}",
                "test_category": "",
            },
        ]
        return [question_sql_templates]
