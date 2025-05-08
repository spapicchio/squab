from typing import TYPE_CHECKING
import pandas as pd
from sqlalchemy import create_engine
from typing_extensions import override
from distilabel.steps import StepInput

from squab.pattern_identification.abstract_pattern_identification import (
    AbstractPatternIdentification,
)
from squab.utils.utils_get_columns_no_pk_no_fk import utils_get_columns_no_pk_fk

if TYPE_CHECKING:
    from distilabel.typing import StepOutput


class PIManyToMany(AbstractPatternIdentification):
    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        count = 0
        engine = create_engine(f"sqlite:///{inputs[0]['table']['db_path']}")
        dataset = []
        for line in inputs:
            table = line["table"]
            column_names = utils_get_columns_no_pk_fk(table)
            pi_metadata = {"pattern_type": "many_to_many"}
            pattern_identification_cost = 0.0
            for i in range(len(column_names) - 1):
                for j in range(i + 1, len(column_names)):
                    col1, col2 = column_names[i], column_names[j]

                    if len(dataset) >= self.max_identified_patterns_per_tbl:
                        self._logger.info(
                            "Maximum number of patterns reached for this table."
                        )
                        break

                    if self._is_many_to_many(
                        table["tbl_name"], col1=col1, col2=col2, engine=engine
                    ):
                        count += 1
                        pattern_identification = {
                            "entity": col1,
                            "component": col2,
                        }
                        line_updated = self.update_line(
                            line,
                            pattern_identification,
                            pattern_identification_cost,
                            pi_metadata,
                        )
                        dataset.append(line_updated)

        yield dataset

    def _is_many_to_many(self, tbl_name: dict, col1: str, col2: str, engine) -> bool:
        # check for multiple unique values in col2 for each unique value in col1
        query = f"SELECT `{col1}`, `{col2}` FROM `{tbl_name}` LIMIT 100;"

        df = pd.read_sql_query(query, engine.connect())

        # check for multiple unique values in col1 for each unique value in col2
        cond1 = any(df.groupby(col1)[col2].nunique() > 1)
        cond2 = any(df.groupby(col2)[col1].nunique() > 1)

        return cond1 and cond2
