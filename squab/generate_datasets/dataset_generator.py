from abc import ABC, abstractmethod
from itertools import islice
from typing import Generator

import pandas as pd
from pydantic import BaseModel
from qatch.connectors import ConnectorTable, SqliteConnector

from .utils import utils_find_closest_matches


class DatasetInput(BaseModel):
    relative_sqlite_db_path: str
    tbl_in_db_to_analyze: list[str] | str | None = None
    db_name: str | None = None
    tables: dict[str, pd.DataFrame] | None = None
    table2primary_key: dict[str, str] | None = None

    max_num_tbls: int | None = 1,
    max_patterns_for_tbl: int | None = 5,
    max_num_metadata_for_pattern: int | None = 5,
    max_questions_for_metadata: int | None = 5


class DatasetGenerator[PatternType, MetadataType, TestType](ABC):
    def generate_dataset(self, function_input: DatasetInput) -> list[TestType]:
        # for loop over the table
        tests = []
        # Apply max_num constraints to each nested loop using `islice`
        for tbl in islice(self.read_table_generator(**function_input.model_dump()), function_input.max_num_tbls):
            for pattern in islice(self.pattern_identification(tbl), function_input.max_patterns_for_tbl):
                for metadata in islice(self.metadata_generator(pattern, **function_input.model_dump()),
                                       function_input.max_num_metadata_for_pattern):
                    for test in islice(self.tests_generator(metadata, **function_input.model_dump()),
                                       function_input.max_questions_for_metadata):
                        tests.append(test)
        return tests

    @abstractmethod
    def pattern_identification(self, table: ConnectorTable) -> Generator[PatternType, None, None]:
        raise NotImplementedError

    @abstractmethod
    def metadata_generator(self, pattern, *args, **kwargs) -> Generator[MetadataType, None, None]:
        raise NotImplementedError

    @abstractmethod
    def tests_generator(self, metadata: MetadataType, *args, **kwargs) -> Generator[TestType, None, None]:
        raise NotImplementedError

    def read_table_generator(self,
                             relative_sqlite_db_path: str,
                             tbl_in_db_to_analyze: list[str] | str | None = None,
                             db_name: str | None = None,
                             tables: dict[str, pd.DataFrame] | None = None,
                             table2primary_key: dict[str, str] | None = None,
                             *args, **kwargs) -> Generator[ConnectorTable, None, None]:
        db_name or relative_sqlite_db_path.split('/')[-1].replace('.sqlite', '')
        sqlite_connector = SqliteConnector(relative_db_path=relative_sqlite_db_path,
                                           db_name=db_name,
                                           tables=tables,
                                           table2primary_key=table2primary_key)
        tbl_name2tbls = sqlite_connector.load_tables_from_database()
        tbl_in_db_to_analyze = utils_find_closest_matches(tbl_in_db_to_analyze, list(tbl_name2tbls.keys()))
        for tbl_name in tbl_in_db_to_analyze:
            yield tbl_name2tbls[tbl_name]
