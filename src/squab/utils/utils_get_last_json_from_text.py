import json
import re

def utils_get_last_json_from_text(text: str) -> dict:
    """
    Extracts the last JSON object from a given text string.

    Args:
        text (str): The input text containing JSON objects.

    Returns:
        dict: The last JSON object found in the text.
    """ 
    # Find all JSON-like patterns in the text
    json_objects = re.findall(r'\{.*?\}', text, re.DOTALL)

    if not json_objects:
        return {}

    # Parse the last JSON object
    last_json_str = json_objects[-1]
    try:
        return json.loads(last_json_str)
    except json.JSONDecodeError:
        return {}