import random
from typing import Literal

from squab.graph_states import Line
from squab.nodes.test_generation.utils_decorator_llm_node import test_generation_based_templates
from squab.nodes.utils import utils_run_qatch, GenerationSteps


@test_generation_based_templates()
def create_templates_vague(
        line: Line, *args, **kwargs
) -> list[list[dict[Literal['test_category', 'query', 'question'], str]]]:
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
    similar_cols = line[GenerationSteps.PI.value]["similar_columns"]
    random.seed(kwargs.get("seed", 42))
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
