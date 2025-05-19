import random
import pandas as pd
from collections import defaultdict
from typing_extensions import override, TYPE_CHECKING
from sqlalchemy import create_engine
from distilabel.steps import StepInput

from squab.pattern_identification.abstract_pattern_identification import (
    AbstractPatternIdentification,
)
from squab.utils.utils_get_columns_no_pk_no_fk import utils_get_columns_no_pk_fk

if TYPE_CHECKING:
    from distilabel.typing import StepOutput


class PIOverlappingCols(AbstractPatternIdentification):
    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        count = 0
        engine = create_engine(f"sqlite:///{inputs[0]['table']['db_path']}")
        dataset = []
        for line in inputs:
            table = line["table"]
            all_columns = utils_get_columns_no_pk_fk(table)
            cat_column_names = utils_get_columns_no_pk_fk(
                table, start_from_cols=list(table["cat_col2metadata"].keys())
            )
            if len(all_columns) < 3:
                self._logger.info(
                    "Not enough columns for pattern identification. Returning pattern_identification=None."
                )
                line_updated = self.update_line(
                    line,
                    None,
                    0.0,
                    0.0,
                )
                dataset.append(line_updated)
                continue

            pi_metadata = {"pattern_type": "overlapping_cols"}
            pattern_identification_cost = 0.0

            column_to_project = random.choice(cat_column_names)

            # Filter columns not related to the projected column
            all_columns_not_selected = [
                col for col in all_columns if col != column_to_project
            ]
            cat_column_names.remove(column_to_project)

            # Iterate through categorical columns and perform matching
            while cat_column_names:
                entity_column = cat_column_names.pop()
                for component_column in all_columns_not_selected:
                    if component_column == entity_column:
                        continue

                    # Check for overlapping columns
                    col1_val1_val2_to_values_col2 = _find_overlapping_column_values(
                        table.tbl_name,
                        entity_column,
                        component_column,
                        engine,
                    )
                    # sample only two overlapping groups in column_1 to avoid explosion
                    sampled_entity_values = random.sample(
                        list(col1_val1_val2_to_values_col2.keys()),
                        min(2, len(col1_val1_val2_to_values_col2)),
                    )

                    for entity_value in sampled_entity_values:
                        pattern_identification = {
                            "entity": entity_column,
                            "component": component_column,
                            "column_to_project": column_to_project,
                            "entity_values": list(entity_value),
                            "component_value": random.choice(
                                col1_val1_val2_to_values_col2[entity_value]
                            ),
                        }
                        line_updated = self.update_line(
                            line,
                            pattern_identification,
                            pattern_identification_cost,
                            pi_metadata,
                        )
                        dataset.append(line_updated)
                        count += 1
                        if len(dataset) >= self.max_identified_patterns_per_tbl:
                            self._logger.info(
                                "Maximum number of patterns reached for this table."
                            )
                            break
        yield dataset

    def _is_many_to_many(self, tbl_name: dict, col1: str, col2: str, engine) -> bool:
        # check for multiple unique values in col2 for each unique value in col1
        query = f"SELECT `{col1}`, `{col2}` FROM `{tbl_name}` LIMIT 100;"

        df = pd.read_sql_query(query, engine.connect())

        # check for multiple unique values in col1 for each unique value in col2
        cond1 = any(df.groupby(col1)[col2].nunique() > 1)
        cond2 = any(df.groupby(col2)[col1].nunique() > 1)

        return cond1 and cond2


def _find_overlapping_column_values(
    table_name: str,
    column1: str,
    column2: str,
    engine,
) -> dict[tuple, list]:
    LIMIT = 50  # Max number of rows to fetch

    # Extracted function to fetch data through query
    def fetch_data(tbl_name: str, col1: str, col2: str) -> list[list]:
        query = f"SELECT `{col1}`, `{col2}` FROM `{tbl_name}` LIMIT {LIMIT};"
        return engine.connect().run(query)

    # Extracted function to build dictionary of column relationships
    def create_column_values_associations(data: list[list]) -> defaultdict:
        # get for each col1_value, the set of col2_values
        value_col12values_col2 = defaultdict(set)
        for col1_value, col2_value in data:
            value_col12values_col2[col1_value].add(col2_value)
        return value_col12values_col2

    # Extracted function to find intersections between column values
    def find_intersections_among_col1_values(
        value_col12values_col2: defaultdict,
    ) -> dict[tuple, list]:
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
