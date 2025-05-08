def is_openai_format(input) -> bool:
    """Checks if the input is in OpenAI chat-like format:

    ```python
    [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
    ]
    ```

    Args:
        input: The input to check.

    Returns:
        A boolean indicating if the input is in OpenAI chat-like format.
    """
    if not isinstance(input, list):
        return False
    return all(
        isinstance(x, dict) and "role" in x.keys() and "content" in x.keys()
        for x in input
    )
