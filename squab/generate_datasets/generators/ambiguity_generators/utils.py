import numpy as np
from sklearn.metrics import pairwise_distances


def utils_combine_clusters(clusters: dict[str, list]) -> dict[str, list]:
    # List of clusters to be removed
    to_remove = set()

    # List of keys for easy iteration
    keys = list(clusters.keys())

    # Iterate through the dictionary
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            if i != j:
                # Check if cluster1 is a subset of cluster2
                if set(clusters[key1]).issubset(set(clusters[key2])):
                    to_remove.add(key1)
                # Check if cluster2 is a subset of cluster1
                elif set(clusters[key2]).issubset(set(clusters[key1])):
                    to_remove.add(key2)

    # Remove the clusters that are subsets
    for key in to_remove:
        del clusters[key]

    return clusters


def utils_get_pairwise_similarity_metric(values: list[list[float]], metric='cosine'):
    pairwise_distance_matrix = pairwise_distances(values, values, metric=metric)
    return 1 - pairwise_distance_matrix


def utils_get_top_k_index_similar_matrix(self, values, at_most_k, threshold) -> list[list[int]]:
    similarity_matrix = utils_get_pairwise_similarity_metric(values, metric='cosine')
    similarity_matrix = np.tril(similarity_matrix)
    # Change all values less than the threshold to 0
    similarity_matrix[similarity_matrix < threshold] = 0
    # Fill the diagonal with 0s to remove same elements
    np.fill_diagonal(similarity_matrix, 0)
    # Use numpy's argsort function to get indices of top-k values along each row
    # This will return a 2D array where each row corresponds to the row in your original array
    # and contains the indices of the top-k values in that row
    top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -at_most_k:]
    # remove the indexes where there is 0 similarity
    top_k_indices = [
        [index for index in attribute_indexes if similarity_matrix[i, index] != 0]
        for i, attribute_indexes in enumerate(top_k_indices)
    ]
    return top_k_indices


import difflib


def utils_syntactic_match(str1: str, str2: str) -> float:
    """
    Returns a similarity ratio between two strings using difflib's SequenceMatcher.

    Args:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.

    Returns:
        float: A similarity ratio between 0 and 1, where 1 means the strings are identical.
    """
    return difflib.SequenceMatcher(None, str1, str2).ratio()
