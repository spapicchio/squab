import random
from typing import Literal

from jinja2 import Template

from squab.graph_states import Line
from squab.nodes.node_llm_call import llm_call
from squab.nodes.utils import utils_run_qatch, utils_get_last_json_from_text

DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant who writes a natural language (NL) question. 
You are provided with a definition of ambiguity, the SQL queries that answer the question following the ambiguity rules, and a database containing the answers. You may also receive metadata helping you in generating the question. Your task is to write the NL question following these guidelines:

- All unformatted table and column names must be replaced with plain words, preferably synonyms.
- Make the question as short as possible, but do not miss any part of the question like order-by (e.g., remove unnecessary words or paraphrase). Yet, you must check the relevant tables to ensure that the question and its interpretations express the same request as the queries and would yield the same answer. Example: You can modify "fitness training program" into "training program" and omit the unnecessary word “fitness” only if "training program"  cannot be confused with other columns in different tables.
- You must maintain ambiguity when writing the question and reading each interpretation.
- If the projected column name can be inferred, remove it from the final output

# Output Format
Provide the answer in JSON format as follows
```json
{
    "question": "the generated question"
}
```
"""

DEFAULT_TEMPLATE = """
## Ambiguity Definition
{{ ambig_definition }}

## queries
{{ queries }}

## Metadata
{{ metadata }}

## Database
{{ database }}
""".rstrip()

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


def process_question_vague(line: Line,
                           litellm_params_vague: dict,
                           vague_user_template: str | None = None,
                           vague_system_template: str | None = None,
                           *args, **kwargs
                           ) -> Line | list[Line]:
    system_prompt = vague_system_template or DEFAULT_SYSTEM_PROMPT
    user_template = vague_user_template or COL_AMB_AMB_DEF
    processed_lines = []
    for qatch_templates in _generate_sql_templates(line):
        sql_interpretations = [
            template["query"] for template in qatch_templates
        ]
        messages = [{"role": "system", "content": system_prompt}] \
                   + COL_AMB_FEW_SHOTS \
                   + [{"role": "user",
                       "content": Template(user_template).render(
                           database=line['db_schema_table_examples'],
                           metadata=line['relational_metadata'],
                           ambig_definition=COL_AMB_AMB_DEF,
                           queries="\n".join(sql_interpretations)
                       )}]

        response, total_cost = llm_call(messages, litellm_params_vague).result()
        model_response = response["choices"][0]["message"]["content"]
        ambig_question = utils_get_last_json_from_text(model_response)
        if not ambig_question:
            line['has_failed'] = {
                'vague': f"The model was not able to generate a vague question. Model Response: {model_response}"
            }
        else:
            line['question'] = ambig_question['question']
            line['target'] = sql_interpretations
            line['test_sub_category'] = qatch_templates[0]['test_category']
            line['qatch_templates'] = qatch_templates

        line['total_cost'] += total_cost
        line['granular_costs']['test_generation'] = total_cost
        processed_lines.append(line)
    return processed_lines


def _generate_sql_templates(line: Line) -> list[list[dict[Literal['test_category', 'query', 'question'], str]]]:
    """
    Generates SQL templates and corresponding questions by replacing columns in the query
    with similar columns, taking into account specific query clauses like SELECT and ORDER BY.

    Args:
        line (Line): A metadata object containing information such as columns, database path,
                     and table name.

    Returns:
        list[list[dict]]: Nested list where each inner list contains variations of one query,
                          with replaced columns and the associated question and test category.
    """
    similar_cols = line["pattern_identification"]["similar_columns"]
    random.seed(42)
    col_in_query = random.choice(similar_cols)
    list_queries_with_selected_col = utils_run_qatch(
        db_path=line['db_path'],
        selected_col=col_in_query,
        tbl_name=line["tbl_name"]
    )

    templates = []

    for test_query in list_queries_with_selected_col:

        query = test_query["query"]
        question = test_query["question"]
        test_category = test_query["test_category"]

        # Generate templates by replacing `col_in_query` with each similar column
        question_sql_templates = [{
            "question": question.replace(col_in_query, col),
            "query": query.replace(col_in_query, col),
            "test_category": test_category,
        } for col in similar_cols]

        # Split query into parts before and after the "FROM" clause
        before_from, after_from = query.lower().split("from", 1)

        # Check if the column is in SELECT but not in aggregation
        if (
                col_in_query in before_from
                and col_in_query not in after_from
                and f"(`{col_in_query}`)" not in query.lower()
        ):
            combined_cols = ", ".join(f"`{col}`" for col in similar_cols)
            question_sql_templates.append(
                {
                    "question": question.replace(col_in_query, combined_cols),
                    "query": query.replace(f"`{col_in_query}`", combined_cols),
                    "test_category": test_category,
                }
            )

        # Check if the column is used in an ORDER BY clause
        if f"order by `{col_in_query}`" in after_from:
            combined_cols = ", ".join(f"`{col}`" for col in similar_cols)
            question_sql_templates.append(
                {
                    "question": question.replace(col_in_query, combined_cols),
                    "query": query.replace(f"`{col_in_query}`", combined_cols),
                    "test_category": test_category,
                }
            )

        templates.append(question_sql_templates)

    return templates
