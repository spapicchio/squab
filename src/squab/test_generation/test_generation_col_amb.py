import random
from qatch.connectors import SqliteConnector
from typing_extensions import override

from squab.test_generation.abstract_test_generation import (
    DEFAULT_TEMPLATE,
    AbstractTestGeneration,
)
from squab.utils.utils_run_qatch import utils_run_qatch

from jinja2 import Template


COL_AMB_AMB_DEF = (
    "Colum Ambiguity arises when a natural language query is insufficiently specific to  identify"
    "a particular column within a table. This ambiguity often occurs when multiple columns "
    "share similar meaning and it is possible to associate these columns to a common label. "
    "As example, consider a table with two columns: `Name` and `Surname`. "
    'A query like "What are the information of Simone?" is ambiguous because '
    "it's uncertain whether the query refers to the Name or the Surname or to both columns. "
    "Given the queries, the semantic similar columns and the label to use in the generation, "
    "generate an ambiguous question that uses the label rather than the columns with the same intent of each "
    "query. Note that you can use also synonyms of the label as long as they are not present in the table. "
)


COL_AMB_FEW_SHOTS = [
    {
        "role": "user",
        "content": Template(DEFAULT_TEMPLATE).render(
            ambig_definition=COL_AMB_AMB_DEF,
            queries="\n".join(
                [
                    "Select Reviews.Hikes, Reviews.customer_review From Reviews",
                    "Select Reviews.Hikes, Reviews.difficulty_level From Reviews",
                    "Select Reviews.Hikes, Reviews.customer_review, difficulty_level From Reviews",
                ]
            ),
            metadata="""{"label": "ratings", "columns": ["customer_review", "difficulty_level"]}""",
            database="CREATE TABLE Reviews (\n   Hikes TEXT,\n   customer_review TEXT,\n   difficulty_level TEXT\n);",
        ),
    },
    {
        "role": "assistant",
        "content": """
        ```json
        {
            "question": "What hikes do we have and what are their ratings?"
        }
        ```
        """,
    },
    {
        "role": "user",
        "content": Template(DEFAULT_TEMPLATE).render(
            ambig_definition=COL_AMB_AMB_DEF,
            queries="\n".join(
                [
                    "SELECT average_years_of_life\r\nFROM LifeExpectancies\r\nORDER BY region_id\r\nLIMIT 1;",
                    "SELECT gender_specific_life_expectancy\r\nFROM LifeExpectancies\r\nORDER BY region_id\r\nLIMIT 1;",
                    "SELECT average_years_of_life, gender_specific_life_expectancy\r\nFROM LifeExpectancies\r\nORDER BY region_id\r\nLIMIT 1;",
                ]
            ),
            metadata="""{"label": "life expectancy", "columns": ["average_years_of_life", "gender_specific_life_expectancy"]}""",
            database="CREATE TABLE LifeExpectancies (\nregion_id TEXT,\n  average_years_of_life TEXT,\n gender_specific_life_expectancy TEXT\n);",
        ),
    },
    {
        "role": "assistant",
        "content": """
        ```json
        {
            "question": "What is the life expectancy of the region with the lowest ID?"
        }
        ```
        """,
    },
]


class TestGeneratorColAmb(AbstractTestGeneration):
    ambiguity_definition: str = COL_AMB_AMB_DEF
    few_shots_messages: list = COL_AMB_FEW_SHOTS

    @override
    def _create_question_sql_templates(self, line):
        similar_cols = line["relational_metadata"]["columns"]
        col_in_query = random.choice(similar_cols)
        sqlite_connector = SqliteConnector(
            db_path=line["table"]["db_path"],
            db_name=line["table"]["db_id"],
            tbl_name=line["table"]["tbl_name"],
        )
        list_queries_with_selected_col = (
            utils_run_qatch(
                sqlite_connector=sqlite_connector,
                selected_col=col_in_query,
                tbl_name=line["table"]["tbl_name"],
            ),
        )

        # Replace the target column with each similar column individually
        for test_category_query_question_dict in list_queries_with_selected_col:
            question_sql_templates = []
            query = test_category_query_question_dict["query"]
            question = test_category_query_question_dict["question"]
            test_category = test_category_query_question_dict["test_category"]
            for col in similar_cols:
                question_sql_templates.append(
                    {
                        "question": question.replace(col_in_query, col),
                        "query": query.replace(col_in_query, col),
                        "test_category": test_category,
                    }
                )
            # Split the query into parts before and after the "FROM" clause
            before_from, after_from = query.lower().split("from", 1)
            # Check if the column is in the SELECT projection but not in an aggregation
            # This is used to also add cases with all columns in the projection or order-by
            if (
                col_in_query in before_from  # Column appears in the SELECT clause
                and col_in_query
                not in after_from  # Column does not appear after "FROM"
                and f"(`{col_in_query}`)"
                not in query.lower()  # Column is not part of an aggregation
            ):
                # Replace the target column with all similar columns, joined by commas
                question_sql_templates.append(
                    {
                        "question": question.replace(
                            col_in_query, ", ".join(f"`{col}`" for col in similar_cols)
                        ),
                        "query": query.replace(
                            f"`{col_in_query}`",
                            ", ".join(f"`{col}`" for col in similar_cols),
                        ),
                        "test_category": test_category,
                    }
                )

            # Check if the column is used in an "ORDER BY" clause
            if f"order by `{col_in_query}`" in after_from:
                # Replace the target column with all similar columns, joined by commas
                question_sql_templates.append(
                    {
                        "question": question.replace(
                            col_in_query, ", ".join(f"`{col}`" for col in similar_cols)
                        ),
                        "query": query.replace(
                            f"`{col_in_query}`",
                            ", ".join(f"`{col}`" for col in similar_cols),
                        ),
                        "test_category": test_category,
                    }
                )
            yield question_sql_templates
