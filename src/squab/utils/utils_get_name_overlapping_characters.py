from rapidfuzz.distance import Levenshtein


def utils_levenshtein_name_in(list_values: list, name: str) -> str:
    """
    Get the name with the most overlapping characters from a list of names.

    Args:
        list_values (list): List of names to compare.
        name (str): Name to compare against.

    Returns:
        str: The name in list_values most similar to name.
    """

    name = name.lower().strip()
    min_distance = float("inf")
    most_overlapping_name = ""
    for val in list_values:
        distance = Levenshtein.distance(val.lower().strip(), name)
        if distance < min_distance:
            min_distance = distance
            most_overlapping_name = val

    return most_overlapping_name
