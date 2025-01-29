import difflib


def utils_find_closest_matches(
        target_words: list[str] | str | None,
        candidate_words: list[str]
) -> list[str]:
    """
    Find the closest matching words from a list of candidate words based on syntactic similarity.

    This function takes a list or a single target word and compares it against a list
    of candidate words, returning the best match for each target word. If no target
    word is provided, the entire candidate word list is returned as is.

    Args:
        target_words (list[str] | str | None): A list of target words, a single
            target word, or None.
        candidate_words (list[str]): A list of candidate words to compare against
            the target.

    Returns:
        list[str]: A list of candidate words that are the closest matches for
            each target word.

    """
    if target_words is None:
        return candidate_words
    if isinstance(target_words, str):
        target_words = [target_words]

    def get_best_match(target: str, candidates: list[str]) -> str:
        scores = [utils_syntactic_match(target, c) for c in candidates]
        return candidates.pop(scores.index(max(scores)))

    return [get_best_match(t, candidate_words) for t in target_words]


def utils_syntactic_match(str1: str, str2: str) -> float:
    """
    Compares two strings syntactically and returns a similarity ratio.

    Uses the SequenceMatcher from the difflib library to determine the
    similarity ratio between two strings based on their syntactic content.

    Args:
        str1 (str): The first string for comparison.
        str2 (str): The second string for comparison.

    Returns:
        float: A floating-point value between 0 and 1 representing the
        similarity ratio. A value of 1 indicates identical strings, while 0
        indicates no similarity.
    """
    return difflib.SequenceMatcher(None, str1, str2).ratio()
