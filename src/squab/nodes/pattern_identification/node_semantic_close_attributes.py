import copy

import litellm

from squab.graph_states import Line
from squab.nodes.utils_decorator_process_dataset import dataset_processor
from squab.nodes.pattern_identification.utils import utils_get_top_k_index_similar_matrix, utils_combine_clusters
from squab.nodes.utils import utils_get_columns_no_pk_fk, GenerationSteps


@dataset_processor(GenerationSteps.PI)
def process_semantic_close_attributes_line(
        line: Line,
        encoder_name: str,
        threshold_similar_values: float,
        *args,
        **kwargs
) -> Line | list[Line]:
    """Process a single line for semantic close attributes identification."""
    columns_no_pk_fk = utils_get_columns_no_pk_fk(line)

    # You need at least two columns to find a pattern
    if len(columns_no_pk_fk) < 2:
        line['has_failed'] = {
            'pi_semantic_close_attributes': "The table has less than two columns excluding primary and foreign keys,"
                                            " cannot find a pattern."
        }
        line['total_cost'] = 0.0
        line['granular_costs']['pattern_identification'] = 0.0
        return line

    # Get similar columns
    similar_columns, total_cost = _get_similar_values(
        columns_no_pk_fk,
        threshold_similar_values=threshold_similar_values,
        encoder_name=encoder_name
    )

    # Process each set of similar columns
    processed_lines = []
    for column_pairs in similar_columns:
        if len(column_pairs) > 1:
            processed_line = copy.deepcopy(line)
            # Update the costs
            processed_line['granular_costs']['pattern_identification'] = total_cost / len(similar_columns)
            processed_line['total_cost'] += total_cost / len(similar_columns)
            processed_line['pattern_identification'] = {"similar_columns": column_pairs}
            processed_lines.append(processed_line)

    # If no similar columns were found, return the original line with a failure marker
    if not processed_lines:
        line['has_failed'] = {
            'pi_semantic_close_attributes': "No similar columns found."
        }
        line['total_cost'] += total_cost
        line['granular_costs']['pattern_identification'] = total_cost
        return line

    return processed_lines


def _get_similar_values(
        values: list[str],
        threshold_similar_values,
        encoder_name,
) -> tuple[list[list[str]], float]:
    at_most_k = int(len(values) / 2)
    at_most_k = 2 if at_most_k < 2 else at_most_k
    parsed_columns = [f'{col}' for col in values]

    vals_embeddings = litellm.embedding(model=encoder_name, input=parsed_columns)
    total_cost = litellm.completion_cost(vals_embeddings)
    vals_embeddings = [val['embedding'] for val in vals_embeddings.data]

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

    return list(cluster_name2similar_cols.values()), total_cost
