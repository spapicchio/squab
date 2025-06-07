import copy
import functools
from typing import Callable, TypeVar, Any

from langgraph.func import task

from squab.graph_states import Line
from squab.nodes.utils import GenerationSteps, utils_check_previous_step

_T = TypeVar("_T")
_TCo = TypeVar("_TCo", covariant=True)


def dataset_processor(step: GenerationSteps):
    """
    Decorator to process a dataset using common operations like:
    - Checking if the previous step was executed
    - Iterating through the dataset
    - Skipping lines that have failed
    - Deep copying and processing each valid line
    """

    def decorator(process_line_fn: Callable[[Line, Any], _T | list[_T]]):
        @functools.wraps(process_line_fn)
        @task
        def wrapper(
                dataset: list[Line],
                *args: Any,
                **kwargs: Any
        ) -> list[_T]:
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

        return wrapper

    return decorator
