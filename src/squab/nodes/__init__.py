from enum import Enum
from typing import Callable

from langgraph.func import task

from src.squab.graph_states import Line
from src.squab.nodes.node_read_sqlite_db import node_read_db_sqlite
from src.squab.nodes.pattern_identification import node_semantic_close_attributes


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
        generator_params: dict
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

    dataset_pi = node_pi(dataset, **generator_params).result()
    dataset_rm = node_rm(dataset_pi, **generator_params).result()
    dataset_tg = node_tg(dataset_rm, **generator_params).result()
    return dataset_tg


def process_dataset_with_generator(dataset: list[Line], generator_name: str, generator_params: dict):
    generator = Generators(generator_name)
    node_pi, node_rm, node_tg = GeneratorsCallable[generator]
    return node_generator(dataset, node_pi=node_pi, node_rm=node_rm, node_tg=node_tg, generator_params=generator_params)
