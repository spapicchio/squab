from typing_extensions import override, TYPE_CHECKING

import litellm
from litellm.cost_calculator import completion_cost
from distilabel.steps import StepInput

from squab.test_generation.abstract_test_generation import (
    DEFAULT_TEMPLATE,
    AbstractTestGeneration,
)
from squab.utils.utils_get_last_json_from_text import utils_get_last_json_from_text

if TYPE_CHECKING:
    from distilabel.typing import StepOutput

from jinja2 import Template


SCOPE_AMB_DEF = (
    "Scope ambiguity occurs when it is unclear how a modifier or phrase is attached to the rest of the sentence. "
    "The ambiguity rise when there is a many-to-many relationship between two columns that have a "
    "'Entity' - 'Component' semantic relation. "
    "Therefore, it is unclear whether the question is asking for all the component present in all the entities "
    "(collective interpretation) or for each entity separately (distributive interpretation). "
    "Consider the NL question 'What activities does each gym offer?' over a table with a many-to-many relationship "
    "between Gym (entity) and Activities (component). "
    "Here, there are two interpretations of the question: in the collective interpretation, "
    "the quantifier is interpreted widely (i.e., 'each gym' refers to all gyms in the database). "
    "Instead, in the distributive interpretation, "
    "the quantifier is interpreted narrowly (i.e., 'each gym' is considered separately)."
)


SCOPE_FEW_SHOTS = [
    {
        "role": "user",
        "content": Template(DEFAULT_TEMPLATE).render(
            ambig_definition=SCOPE_AMB_DEF,
            queries="SELECT ClassName FROM GymClasses GROUP BY ClassID, ClassName HAVING COUNT(DISTINCT GymID) = (SELECT COUNT(DISTINCT GymID) FROM GymClasses);\nSELECT DISTINCT GymName, ClassName FROM GymClasses ",
            metadata="""{"entity": "Gym", "component": "GymClass"}""",
            database="CREATE TABLE GymClasses (\n   GymID INTEGER,\n    ClassID INTEGER,\n    ClassName TEXT,\n    GymName TEXT,\n    Location TEXT,\n    MembershipType TEXT\n);",
        ),
    },
    {
        "role": "assistant",
        "content": """
        ```json
        {
            "question": "What activities does each gym offer?"
        } 
        """,
    },
    {
        "role": "user",
        "content": Template(DEFAULT_TEMPLATE).render(
            ambig_definition=SCOPE_AMB_DEF,
            queries="SELECT `Genre` FROM `Movies` GROUP BY `Genre` HAVING COUNT(DISTINCT `Budget`) = (SELECT COUNT(DISTINCT `Budget`) FROM `Movies`);\nSELECT DISTINCT `Genre`, `Budget` FROM `Movies`",
            metadata="""{"entity": "Movies", "component": "Genre"}""",
            database="CREATE TABLE Movies (\n   Genre TEXT,\n    Budget INTEGER,\n    ReleaseYear INTEGER,\n    Director TEXT\n);",
        ),
    },
    {
        "role": "assistant",
        "content": """
        ```json
        {
            "question": "List movie genres associated with the budgets of each movie."
        } 
        """,
    },
]


class TestGeneratorScope(AbstractTestGeneration):
    ambiguity_definition: str = SCOPE_AMB_DEF
    few_shots_messages: list = SCOPE_FEW_SHOTS

    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        dataset = []
        for line in inputs:
            tbl_name = line["table"]["tbl_name"]
            entity = line["relational_metadata"]["entity"]
            component = line["relational_metadata"]["component"]
            sql_interpretations, question_sql_templates = (
                self._build_sql_interpretations(entity, component, tbl_name)
            )

            messages = self.get_messageges_with_user_question(
                ambig_definition=self.ambiguity_definition,
                queries="\n".join(sql_interpretations),
                metadata=str(line["relational_metadata"]),
                database=str(line["table"]["db_schema"]),
            )

            response = litellm.completion(
                model=self.model_name,
                messages=messages,
            )

            test_metadata = response.model_dump()
            test_metadata["messages"] = messages
            test_metadata["question_sql_templates"] = question_sql_templates
            test_metadata["ambig_temp_question"] = "To be implemented"

            test_cost = completion_cost(completion_response=response)

            test_question = utils_get_last_json_from_text(
                response["choices"][0]["message"]["content"]
            )
            test_target = sql_interpretations

            if len(test_question) > 0:
                line = self.update_line(
                    line_to_update=line,
                    test_question=test_question,
                    test_target=test_target,
                    test_cost=test_cost,
                    test_metadata=test_metadata,
                )
                dataset.append(line)
        yield dataset

    def _build_sql_interpretations(self, entity, component, tbl_name):
        query_1 = (
            f"SELECT `{component}` "
            f"FROM `{tbl_name}` "
            f"GROUP BY `{component}` "
            f"HAVING COUNT(DISTINCT `{entity}`) = (SELECT COUNT(DISTINCT `{entity}`) FROM `{tbl_name}`)"
        )
        question_1 = "question_1"

        query_2 = f"SELECT DISTINCT `{component}`, `{entity}` FROM `{tbl_name}` "
        question_2 = "question_2"
        question_sql_templates = [
            {
                "question": question_1,
                "sql_template": query_1,
            },
            {
                "question": question_2,
                "sql_template": query_2,
            },
        ]
        return [query_1, query_2], question_sql_templates
