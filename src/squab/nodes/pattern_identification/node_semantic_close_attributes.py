import copy
import os

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAIEmbeddings
from langgraph.func import task

from squab.graph_states import Line
from squab.nodes.generation_steps import GenerationSteps
from squab.nodes.pattern_identification.utils import utils_get_top_k_index_similar_matrix, utils_combine_clusters
from squab.nodes.utils import utils_check_previous_step, utils_get_columns_no_pk_fk

load_dotenv(override=True)


@task
def node_semantic_close_attributes(
        dataset: list[Line],
        encoder_name: str,
        threshold_similar_values: float,
        *args,
        **kwargs
) -> list[Line]:
    # check if the previous step has been executed
    utils_check_previous_step(dataset, GenerationSteps.PI)
    encoder = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv('OPENAI_API_KEY')
    )
    processed_dataset = []
    for line in dataset:
        lines = copy.deepcopy(line)
        if 'has_failed' not in lines:
            columns_no_pk_fk = utils_get_columns_no_pk_fk(lines)
            # you need at least two columns to find a pattern
            if len(columns_no_pk_fk) < 2:
                lines['has_failed'] = {
                    'pi_semantic_close_attributes': "The table has less than two columns excluding primary and foreign keys,"
                                                    " cannot find a pattern."
                }
            else:
                lines = _process_line(lines, columns_no_pk_fk, encoder, threshold_similar_values)

        lines = lines if isinstance(lines, list) else [lines]
        processed_dataset.extend(lines)
    return processed_dataset


def _process_line(line: Line, columns_no_pk_fk, encoder, threshold_similar_values) -> list[Line]:
    with get_openai_callback() as cb:
        similar_columns = _get_similar_values(
            columns_no_pk_fk,
            threshold_similar_values=threshold_similar_values,
            encoder=encoder
        )
    processed_lines = []
    for column_pairs in similar_columns:
        if len(column_pairs) > 1:
            processed_line = copy.deepcopy(line)
            # update the costs
            processed_line['granular_costs']['pattern_identification'] = cb.total_cost
            processed_line['total_cost'] += cb.total_cost
            processed_line['pattern_identification'] = {"similar_columns": column_pairs}
            processed_lines.append(processed_line)
    return processed_lines


def _get_similar_values(
        values: list[str],
        threshold_similar_values,
        encoder,
) -> list[list[str]]:
    at_most_k = int(len(values) / 2)
    at_most_k = 2 if at_most_k < 2 else at_most_k
    parsed_columns = [f'{col}' for col in values]

    vals_embeddings = encoder.embed_documents(parsed_columns)

    top_k_indexes = utils_get_top_k_index_similar_matrix(vals_embeddings,
                                                         at_most_k=at_most_k,
                                                         threshold=threshold_similar_values)

    cluster_name2similar_cols = {
        f'cluster_{i}': [values[j] for j in indexes] + [values[i]]
        for i, indexes in enumerate(top_k_indexes)
        if indexes
    }
    # combine the values which are subsets of each other
    cluster_name2similar_cols = utils_combine_clusters(cluster_name2similar_cols)

    return list(cluster_name2similar_cols.values())
