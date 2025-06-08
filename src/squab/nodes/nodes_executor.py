from enum import Enum
from typing import Any

from langgraph.func import task

from squab.graph_states import Line
from squab.logger import get_logger
from squab.nodes.pattern_identification import (
    process_semantic_close_attributes_line,
    get_many_to_many_from_line,
    get_overlapping_cols
)
from squab.nodes.relational_metadata import (
    process_hypernym_line,
    find_entity_component_from,
    create_new_attributes_from
)
from squab.nodes.test_generation import (
    create_templates_vague,
    create_templates_scope,
    create_templates_attach,
    create_templates_col_unans
)


class CategoryType(Enum):
    SCOPE = "scope"
    ATTACHMENT = "attachment"
    VAGUE = "vague"
    COL_UNANS = "col_unans"
    CALC_UNANS = "calc_unans"
    OUT_OF_SCOPE = "out_of_scope"


category_handlers = {
    CategoryType.ATTACHMENT: [get_overlapping_cols, create_templates_attach],
    CategoryType.SCOPE: [get_many_to_many_from_line, find_entity_component_from, create_templates_scope],
    CategoryType.VAGUE: [process_semantic_close_attributes_line, process_hypernym_line, create_templates_vague],
    CategoryType.COL_UNANS: [create_new_attributes_from, create_templates_col_unans],
    CategoryType.CALC_UNANS: [],
    CategoryType.OUT_OF_SCOPE: [],
}


@task
def execute_generator_handlers(
        dataset: list[Line],
        *generator_handlers,
        generator_params: dict,
        logger: Any
):
    """
    """
    logger.info(f"Selected handlers: {[name.__name__ for name in generator_handlers]}")
    logger.info(f"Generators parameters: {generator_params}")
    processed_dataset = dataset
    for generator_handler in generator_handlers:
        logger.info(f"Starting: `{generator_handler.__name__}`")
        logger.info(f"Dataset size before processing: {len(dataset)}")
        processed_dataset = generator_handler(processed_dataset, logger=logger, **generator_params).result()
        logger.info(f"Dataset size after processing: {len(processed_dataset)}")
    return processed_dataset


def process_dataset_with_generator(dataset: list[Line], generator_name: str, generator_params: dict):
    generator = CategoryType(generator_name)
    generator_handlers = category_handlers[generator]
    for line in dataset:
        line["task_type"] = generator_name
    logger = get_logger(generator_name)
    return execute_generator_handlers(dataset, *generator_handlers, generator_params=generator_params, logger=logger)
