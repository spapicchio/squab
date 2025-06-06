import copy
from typing import Callable, Any

from langgraph.func import task

from squab.graph_states import Line
from squab.nodes.utils import GenerationSteps, T, utils_check_previous_step


@task
def node_process_dataset(
        dataset: list[Line],
        step: GenerationSteps,
        process_line_fn: Callable[[Line, Any], T | list[T]],
        *args: Any,
        **kwargs: Any
) -> list[T]:
    """
    Higher-order function that handles the common pattern in node functions:
    - Checking if the previous step was executed
    - Iterating through the dataset
    - Skipping lines that have failed
    - Creating a deep copy of each line
    - Processing each valid line with the provided function
    - Returning the processed dataset

    Args:
        dataset (list[Line]): The dataset to process
        step (squab.nodes.GenerationSteps): The current step, used to check if the previous step was executed
        process_line_fn (Callable): Function that processes a single line and returns either a single result or a list of results
        *args, **kwargs: Additional arguments to pass to the processing function

    Returns:
        list: The processed dataset
    """
    # Check if the previous step has been executed
    utils_check_previous_step(dataset, step)
    processed_dataset = []

    for line in dataset:
        # Skip lines that have failed
        if 'has_failed' in line:
            processed_dataset.append(line)
            continue

        # Create a deep copy to avoid modifying the original
        line_copy = copy.deepcopy(line)

        # Process the line with the provided function
        result = process_line_fn(line_copy, *args, **kwargs)

        # Handle both single results and lists of results
        if isinstance(result, list):
            processed_dataset.extend(result)
        else:
            processed_dataset.append(result)

    return processed_dataset
