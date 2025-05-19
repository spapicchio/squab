from typing_extensions import override, Literal

from squab.test_generation.abstract_test_generation import (
    DEFAULT_TEMPLATE,
    AbstractTestGeneration,
)


from jinja2 import Template

TOKEN_TYPE_PROMPT = (
    "Type/token ambiguity occurs when a term in a natural language question can refer both to a general category "
    "(type) and to individual instances (tokens) of that category. "
    "This leads to multiple plausible SQL interpretations depending on whether the question asks for the total number of "
    "individual records (tokens) or for the number of distinct categories (types). "
    "For example, consider a table that logs car sales with a 'Model' column. The question 'How many cars did we sell in 2024?' "
    "can be interpreted in two ways: one asking for the total number of cars sold (token interpretation, using COUNT(*)) "
    "and one asking for the number of distinct models sold (type interpretation, using COUNT(DISTINCT Model))."
)


TOKEN_TYPE_FEW_SHOTS = [
    {
        "role": "user",
        "content": Template(DEFAULT_TEMPLATE).render(
            ambig_definition=TOKEN_TYPE_PROMPT,
            queries="SELECT COUNT(*) FROM CarSales WHERE SaleYear = 2024;\nSELECT COUNT(DISTINCT Model) FROM CarSales WHERE SaleYear = 2024;",
            metadata="""{"type_column": "Model"}""",
            database="CREATE TABLE CarSales (\n   CarID INTEGER,\n   Model TEXT,\n   SaleYear INTEGER,\n   Price INTEGER\n);",
        ),
    },
    {
        "role": "assistant",
        "content": """
        ```json
        {
            "question": "How many cars did we sell in 2024?"
        }
        ```
        """,
    },
    {
        "role": "user",
        "content": Template(DEFAULT_TEMPLATE).render(
            ambig_definition=TOKEN_TYPE_PROMPT,
            queries="SELECT COUNT(*) FROM BookSales;\nSELECT COUNT(DISTINCT Title) FROM BookSales;",
            metadata="""{"type_column": "Title"}""",
            database="CREATE TABLE BookSales (\n   BookID INTEGER,\n   Title TEXT,\n   Author TEXT,\n   SaleYear INTEGER\n);",
        ),
    },
    {
        "role": "assistant",
        "content": """
        ```json
        {
            "question": "How many books were sold?"
        }
        ```
        """,
    },
]


class TestGeneratorTokenType(AbstractTestGeneration):
    ambiguity_definition: str = TOKEN_TYPE_PROMPT
    few_shots_messages: list = TOKEN_TYPE_FEW_SHOTS

    @override
    def _create_question_sql_templates(
        self, line
    ) -> list[dict[Literal["query", "question", "test_category"]], str]:
        
        type_col_name = line["relational_metadata"]
        ambiguous_term = line["relational_metadata"]["ambiguous_term"]

        pattern_identification = line["pattern_identification"]
        tbl_name = line["table"]["tbl_name"]

        query_1 = f"SELECT COUNT(*) FROM `{tbl_name}`"
        question_1 = f"How many individual {ambiguous_term} are present in the table?"

        query_2 = f"SELECT COUNT(DISTINCT `{pattern_identification}`) FROM `{tbl_name}`"
        question_2 = f"How many distinct {ambiguous_term} {type_col_name} are present in the table?"

        question_sql_templates = [
            {
                "question": question_1,
                "query": query_1,
                "test_category": "token reading",
            },
            {
                "question": question_2,
                "query": query_2,
                "test_category": "type reading",
            },
        ]
        return [question_sql_templates]
