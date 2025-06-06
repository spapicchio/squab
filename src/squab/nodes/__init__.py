from enum import Enum
from typing import Callable, Any

from langgraph.func import task

from squab.graph_states import Line
from squab.logger import get_logger
from squab.nodes.node_process_dataset import node_process_dataset
from squab.nodes.node_read_sqlite_db import node_read_db_sqlite
from squab.nodes.pattern_identification import process_semantic_close_attributes_line
from squab.nodes.relational_metadata import process_hypernym_line
from squab.nodes.test_generation import process_question_vague
from squab.nodes.utils import GenerationSteps, T, utils_check_previous_step


class CategoryType(Enum):
    SCOPE = "scope"
    ATTACHMENT = "attachment"
    VAGUE = "vague"
    COL_UNANS = "col_unans"
    CALC_UNANS = "calc_unans"
    OUT_OF_SCOPE = "out_of_scope"


category_handlers = {
    CategoryType.SCOPE: [],
    CategoryType.ATTACHMENT: [],
    CategoryType.VAGUE: [process_semantic_close_attributes_line, process_hypernym_line, process_question_vague],
    CategoryType.COL_UNANS: [],
    CategoryType.CALC_UNANS: [],
    CategoryType.OUT_OF_SCOPE: [],
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
    dataset_pi = node_process_dataset(dataset,
                                      step=GenerationSteps.PI,
                                      process_line_fn=node_pi,
                                      **generator_params).result()

    logger.info(f"Dataset size after processing: {len(dataset_pi)}")
    dataset_rm = node_process_dataset(dataset_pi,
                                      step=GenerationSteps.RM,
                                      process_line_fn=node_rm,
                                      **generator_params).result()

    logger.info(f"Dataset size after relational metadata: {len(dataset_rm)}")
    dataset_tg = node_process_dataset(dataset_rm,
                                      step=GenerationSteps.TG,
                                      process_line_fn=node_tg,
                                      **generator_params).result()

    logger.info(f"Dataset size after test generation: {len(dataset_tg)}")
    logger.info(f"Finished processing dataset with: {node_pi.__name__}, {node_rm.__name__}, {node_tg.__name__}")
    return dataset_tg


def process_dataset_with_generator(dataset: list[Line], generator_name: str, generator_params: dict):
    generator = CategoryType(generator_name)
    node_pi, node_rm, node_tg = category_handlers[generator]
    for line in dataset:
        line["task_type"] = generator_name
    logger = get_logger(generator_name)
    return node_generator(dataset, node_pi=node_pi, node_rm=node_rm, node_tg=node_tg, generator_params=generator_params,
                          logger=logger)
