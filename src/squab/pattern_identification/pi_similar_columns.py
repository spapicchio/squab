import litellm
from typing import TYPE_CHECKING
from typing_extensions import override
from distilabel.steps import StepInput


from squab.pattern_identification.abstract_pattern_identification import (
    AbstractPatternIdentification,
)
from squab.pattern_identification.utils import (
    utils_combine_clusters,
    utils_get_top_k_index_similar_matrix,
)
from squab.utils.utils_get_columns_no_pk_no_fk import utils_get_columns_no_pk_fk

if TYPE_CHECKING:
    from distilabel.typing import StepOutput


class PISimilarColumns(AbstractPatternIdentification):
    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        count = 0
        dataset = []

        for line in inputs:
            table = line["table"]
            column_names = utils_get_columns_no_pk_fk(table)
            pi_metadata: dict[str, str] = {"pattern_type": "similar_columns"}
            similar_columns, cost = self._get_similar_columns_and_cost(
                columns=column_names,
                threshold_similar_values=0.60,
            )
            for similar_cols in similar_columns:
                line_updated = self.update_line(
                    line,
                    pattern_identification=similar_cols,
                    pattern_identification_cost=cost,
                    pi_metadata=pi_metadata,
                )
                dataset.append(line_updated)
                count += 1
                if count >= self.max_identified_patterns_per_tbl:
                    self._logger.info(
                        "Maximum number of patterns reached for this table."
                    )
                    break
        yield dataset

    def _get_similar_columns_and_cost(
        self,
        columns: list[str],
        threshold_similar_values,
    ) -> list[list[str]]:
        at_most_k = int(len(columns) / 2)
        at_most_k = 2 if at_most_k < 2 else at_most_k

        response = litellm.embedding(input=columns, model="text-embedding-3-small")
        pi_cost = litellm.cost_calculator.completion_cost(completion_response=response)

        # get embeddings of all columns
        vals_embeddings = [embedding['embedding'] for embedding in response.data]

        # build similarity matrix
        top_k_indexes = utils_get_top_k_index_similar_matrix(
            vals_embeddings, at_most_k=at_most_k, threshold=threshold_similar_values
        )

        cluster_name2similar_cols = {
            f"cluster_{i}": [columns[j] for j in indexes] + [columns[i]]
            for i, indexes in enumerate(top_k_indexes)
            if indexes
        }
        # combine the values which are subsets of each other
        cluster_name2similar_cols = utils_combine_clusters(cluster_name2similar_cols)

        return [
            [col for col in similar_cols]
            for similar_cols in cluster_name2similar_cols.values()
        ], pi_cost
