from enum import Enum
from typing import Callable, Any

from langgraph.func import task

from squab.graph_states import Line
from squab.logger import get_logger
from squab.nodes.node_read_sqlite_db import node_read_db_sqlite
from squab.nodes.pattern_identification import node_semantic_close_attributes


class Generators(Enum):
    SCOPE = "scope"
    ATTACHMENT = "attachment"
    VAGUE = "vague"
    COL_UNANS = "col_unans"
    CALC_UNANS = "calc_unans"
    OUT_OF_SCOPE = "out_of_scope"


GeneratorsCallable = {
    Generators.SCOPE: [],
    Generators.ATTACHMENT: [],
    Generators.VAGUE: [node_semantic_close_attributes, node_semantic_close_attributes, node_semantic_close_attributes],
    Generators.COL_UNANS: [],
    Generators.CALC_UNANS: [],
    Generators.OUT_OF_SCOPE: [],
}


@task
def node_generator(
        dataset: list[Line],
        node_pi: Callable,
        node_rm: Callable,
        node_tg: Callable,
        generator_params: dict,
        logger: Any
):
    """
    Generate a node for handling column ambiguity in the graph.

    This function creates a node that checks for column ambiguity based on the previous steps
    of pattern identification, relational metadata, and table generation.

    Args:
        node_pi: Node for pattern identification.
        node_rm: Node for relational metadata.
        node_tg: Node for table generation.

    Returns:
        A function that processes the dataset to identify column ambiguities.
    """
    logger.info(f"Processing dataset with generator: {node_pi.__name__}, {node_rm.__name__}, {node_tg.__name__}")
    logger.info(f"Generator parameters: {generator_params}")
    logger.info(f"Dataset size before processing: {len(dataset)}")
    # Run the pattern identification step
    dataset_pi = node_pi(dataset, **generator_params).result()
    logger.info(f"Dataset size after processing: {len(dataset_pi)}")
    dataset_rm = node_rm(dataset_pi, **generator_params).result()
    logger.info(f"Dataset size after relational metadata: {len(dataset_rm)}")
    dataset_tg = node_tg(dataset_rm, **generator_params).result()
    logger.info(f"Dataset size after table generation: {len(dataset_tg)}")
    logger.info(f"Finished processing dataset with: {node_pi.__name__}, {node_rm.__name__}, {node_tg.__name__}")
    return dataset_tg


def process_dataset_with_generator(dataset: list[Line], generator_name: str, generator_params: dict):
    generator = Generators(generator_name)
    node_pi, node_rm, node_tg = GeneratorsCallable[generator]
    for line in dataset:
        line["task_type"] = generator_name
    logger = get_logger(generator_name)
    return node_generator(dataset, node_pi=node_pi, node_rm=node_rm, node_tg=node_tg, generator_params=generator_params,
                          logger=logger)
