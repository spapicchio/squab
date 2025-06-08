import copy
import random
from collections import defaultdict

from sqlalchemy import create_engine, text

from squab.graph_states import Line
from squab.nodes.utils import utils_get_columns_no_pk_fk, GenerationSteps
from squab.nodes.utils_decorator_process_dataset import dataset_processor


@dataset_processor()
def get_overlapping_cols(
        line: Line,
        *args,
        **kwargs
):
    # Get columns excluding primary and foreign keys
    non_key_columns = utils_get_columns_no_pk_fk(line)

    # Get categorical columns from metadata
    categorical_columns = utils_get_columns_no_pk_fk(
        line, start_from_cols=list(line["cat_col2metadata"].keys())
    )

    # Initialize costs
    line['total_cost'] += 0.0
    line['granular_costs']['pattern_identification'] = 0.0

    if len(non_key_columns) < 3:  # Ensure there are enough columns to process
        line['has_failed'] = {
            GenerationSteps.PI.value: "Not enough columns to identify overlapping columns, needed at least 3."
        }
        return line

    # Initialize database engine
    engine = create_engine(f"sqlite:///{line['db_path']}")

    # Select a random categorical column to project
    random.seed(kwargs.get("seed", 42))
    column_to_project = random.choice(categorical_columns)

    # Exclude the projected column from the others
    non_selected_columns = [col for col in non_key_columns if col != column_to_project]
    categorical_columns.remove(column_to_project)

    # Identify patterns based on overlapping column values
    identified_patterns = _identify_patterns_for_columns(
        line['tbl_name'], categorical_columns, non_selected_columns, column_to_project, engine
    )

    # Create new lines with identified patterns or handle failure case
    lines = []
    for pattern in identified_patterns:
        line[GenerationSteps.PI.value] = pattern
        lines.append(copy.deepcopy(line))

    if not lines:  # No patterns found
        line['has_failed'] = {
            GenerationSteps.PI.value: "No overlapping columns found."
        }
        lines.append(line)

    return lines


def _identify_patterns_for_columns(table_name: str, categorical_columns: list[str],
                                   non_selected_columns: list[str], column_to_project: str, engine) -> list[dict]:
    """Identifies overlapping column patterns and returns the results."""
    identified_patterns = []

    while categorical_columns:  # Iterate through categorical columns
        entity_column = categorical_columns.pop()
        for component_column in non_selected_columns:
            if component_column == entity_column:  # Skip if the columns are the same
                continue
            # Check for overlapping column values
            col1_val1_val2_to_values_col2 = _find_overlapping_column_values(
                table_name, entity_column, component_column, engine
            )

            # Sample up to two overlapping groups to avoid results explosion
            sampled_entity_values = random.sample(
                list(col1_val1_val2_to_values_col2.keys()),
                min(2, len(col1_val1_val2_to_values_col2))
            )

            for entity_value in sampled_entity_values:
                identified_patterns.append({
                    "entity": entity_column,
                    "component": component_column,
                    "column_to_project": column_to_project,
                    "entity_values": list(entity_value),
                    "component_value": random.choice(
                        col1_val1_val2_to_values_col2[entity_value]
                    ),
                })

    return identified_patterns


def _find_overlapping_column_values(
        table_name: str,
        column1: str,
        column2: str,
        engine,
) -> dict[tuple, list]:
    """
    Identifies overlapping values between two columns in a database table.

    For each pair of distinct values in `column1`, finds the intersection of their associated values in `column2`.
    Returns a dictionary mapping each tuple of two `column1` values to the list of overlapping `column2` values.

    Args:
        table_name (str): Name of the table to query.
        column1 (str): The first column to analyze.
        column2 (str): The second column to analyze.
        engine: SQLAlchemy engine for database connection.

    Returns:
        dict[tuple, list]: Mapping of (col1_value1, col1_value2) to list of overlapping col2 values.
    """
    LIMIT = 50  # Max number of rows to fetch

    def fetch_data(tbl_name: str, col1: str, col2: str) -> list[list]:
        """
        Fetches up to LIMIT rows of two columns from the specified table.

        Args:
            tbl_name (str): Table name.
            col1 (str): First column name.
            col2 (str): Second column name.

        Returns:
            list[list]: List of [col1_value, col2_value] pairs.
        """
        query = f"SELECT `{col1}`, `{col2}` FROM `{tbl_name}` LIMIT {LIMIT}"
        return engine.connect().execute(text(query)).fetchall()

    def create_column_values_associations(data: list[list]) -> defaultdict:
        """
        Builds a mapping from each value in col1 to the set of associated col2 values.

        Args:
            data (list[list]): List of [col1_value, col2_value] pairs.

        Returns:
            defaultdict: Mapping from col1_value to set of col2_values.
        """
        value_col12values_col2 = defaultdict(set)
        for col1_value, col2_value in data:
            value_col12values_col2[col1_value].add(col2_value)
        return value_col12values_col2

    def find_intersections_among_col1_values(
            value_col12values_col2: defaultdict,
    ) -> dict[tuple, list]:
        """
        Finds intersections of col2 values for each pair of distinct col1 values.

        Args:
            value_col12values_col2 (defaultdict): Mapping from col1_value to set of col2_values.

        Returns:
            dict[tuple, list]: Mapping of (col1_value1, col1_value2) to list of overlapping col2 values.
        """
        col1_val1_val2_to_values_col2 = {}
        col1_values = list(value_col12values_col2.keys())
        for i in range(len(col1_values)):
            col1_value1 = col1_values[i]
            if "'" in str(col1_value1):
                continue
            for j in range(i + 1, len(col1_values)):
                col1_value2 = str(col1_values[j])
                if "'" in col1_value2:
                    continue

                intersection_values_col2 = value_col12values_col2[
                    col1_value1
                ].intersection(value_col12values_col2[col1_value2])
                intersection_values_col2 = [
                    val for val in intersection_values_col2 if "'" not in str(val)
                ]
                if intersection_values_col2:
                    col1_val1_val2_to_values_col2[(col1_value1, col1_value2)] = list(
                        intersection_values_col2
                    )
        return col1_val1_val2_to_values_col2

    # Main function logic
    data = fetch_data(table_name, column1, column2)
    column_relations = create_column_values_associations(data)
    overlapping_pairs = find_intersections_among_col1_values(column_relations)

    return overlapping_pairs
