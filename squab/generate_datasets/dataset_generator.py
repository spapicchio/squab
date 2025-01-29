import random
from abc import ABC, abstractmethod
from itertools import islice
from typing import Generator
from typing import Optional, Union

import pandas as pd
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel, Field
from qatch.connectors import ConnectorTable, SqliteConnector

from .utils import utils_find_closest_matches


class DatasetInput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    relative_sqlite_db_path: str = Field(description="Relative path to the SQLite database.")
    tbl_in_db_to_analyze: Optional[Union[list[str], str]] = Field(
        None,
        description="Name(s) of the specific tables to analyze in the database. Can be a string or a list of strings.",
    )
    db_name: Optional[str] = Field(None, description="Database name. Extracted from the SQLite path if not provided.")
    tables: Optional[dict[str, pd.DataFrame]] = Field(
        None,
        description="Preloaded tables as a dictionary with table names as keys and DataFrames as values.",
    )
    table2primary_key: Optional[dict[str, str]] = Field(
        None,
        description="Mapping of table names to their primary keys.",
    )
    max_num_tbls: Optional[int] = Field(
        1,
        description="Maximum number of tables to process per database.",
        ge=1,  # Ensure the value is greater than or equal to 1.
    )
    max_patterns_for_tbl: Optional[int] = Field(
        5,
        description="Maximum number of patterns to analyze per table.",
        ge=1,  # Ensure the value is greater than or equal to 1.
    )
    max_num_metadata_for_pattern: Optional[int] = Field(
        5,
        description="Maximum number of metadata entries per pattern.",
        ge=1,  # Ensure the value is greater than or equal to 1.
    )
    max_questions_for_metadata: Optional[int] = Field(
        5,
        description="Maximum number of questions to generate per metadata entry.",
        ge=1,  # Ensure the value is greater than or equal to 1.
    )


class DatasetGenerator[PatternType, MetadataType, TestType](ABC):
    """
    Abstract base class for generating datasets through analysis of tables, patterns, metadata,
    and tests.

    The class serves as a blueprint for creating dataset generators, providing methods to process
    databases, identify patterns, generate metadata, and produce test cases. Subclasses should
    implement the abstract methods for pattern identification, metadata generation, and test
    generation as per the requirements of the dataset. Additionally, utility methods such as reading
    tables and filtering columns are provided to assist in dataset creation.

    Attributes:
        seed (int): Random seed for ensuring reproducibility when generating datasets.
    """

    def __init__(self, seed):
        random.seed(seed)
        self.seed = seed

    def generate_dataset(self, function_input: DatasetInput) -> list[TestType]:
        """
        Generates a dataset of tests by iterating over tables, patterns, metadata, and tests.

        This method uses input constraints such as the maximum number of tables, patterns, metadata, 
        and questions to generate a comprehensive set of tests efficiently. The final dataset is 
        constructed by applying nested loops along with slicing for each generator step, adhering to 
        the provided constraints.

        Args:
            function_input (DatasetInput): The input configuration containing the maximum limits 
                and necessary settings for generating the dataset.

        Returns:
            list[TestType]: A list of generated tests based on the constraints and input provided.
        """
        # for loop over the table
        tests = []

        db_name = function_input.db_name or function_input.relative_sqlite_db_path.split('/')[-1].replace('.sqlite', '')
        sqlite_connector = SqliteConnector(relative_db_path=function_input.relative_sqlite_db_path,
                                           db_name=db_name,
                                           tables=function_input.tables,
                                           table2primary_key=function_input.table2primary_key)

        # Apply max_num constraints to each nested loop using `islice`
        with get_openai_callback() as cb:
            for tbl in islice(
                    self.read_table_generator(sqlite_connector, **function_input.model_dump()),
                    function_input.max_num_tbls
            ):
                for pattern in islice(
                        self.pattern_identification(tbl,
                                                    sqlite_connector=sqlite_connector),
                        function_input.max_patterns_for_tbl
                ):
                    # TODO pass as argument the max_num_metadata_for_pattern to improve generation
                    for metadata in islice(
                            self.metadata_generator(pattern,
                                                    table=tbl,
                                                    sqlite_connector=sqlite_connector),
                            function_input.max_num_metadata_for_pattern
                    ):
                        for test in islice(
                                self.tests_generator(metadata,
                                                     pattern=pattern,
                                                     table=tbl,
                                                     sqlite_connector=sqlite_connector),
                                function_input.max_questions_for_metadata
                        ):
                            tests.append(test)
        # TODO: add cost in test_type
        return tests

    @abstractmethod
    def pattern_identification(self, table: ConnectorTable, *args, **kwargs) -> Generator[PatternType, None, None]:
        """
        Identifies specific patterns within the provided table and generates results.

        This is an abstract method and must be implemented by subclasses. It is designed to
        process the given `ConnectorTable` instance, identify specific patterns contained,
        and yield them as a generator of `PatternType`.

        Args:
            table (ConnectorTable): The data structure used for storing and processing 
                the table information where the method will perform pattern recognition.
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.

        Yields:
            PatternType: Objects representing identified patterns within the provided 
                table.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def metadata_generator(self, pattern: PatternType, *args, **kwargs) -> Generator[MetadataType, None, None]:
        """
        Generates metadata objects based on the provided pattern and arguments. This method is abstract
        and must be implemented by subclasses. It yields metadata objects of type `MetadataType` in a
        generator format.

        Args:
            pattern: Determines the structure or format for metadata generation.
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        Yields:
            MetadataType: Generated metadata objects based on the specified pattern
            and arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def tests_generator(self, metadata: MetadataType, *args, **kwargs) -> Generator[TestType, None, None]:
        """
        Generates test cases based on the provided metadata and additional arguments. This is an
        abstract method that must be implemented by subclasses to define the specific behavior of
        test case generation.

        Args:
            metadata (MetadataType): Metadata details required for generating the tests.
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.

        Yields:
            TestType: Yields generated test cases of type `TestType`.

        Raises:
            NotImplementedError: Raised when the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def read_table_generator(self,
                             sqlite_connector: SqliteConnector,
                             tbl_in_db_to_analyze: list[str] | str | None = None,
                             *args, **kwargs) -> Generator[ConnectorTable, None, None]:
        """
        Yields tables from the database that match the given table names provided or the closest matching ones.

        Args:
            sqlite_connector (SqliteConnector): An instance of the SqliteConnector class used to interact with
                the SQLite database.
            tbl_in_db_to_analyze (list[str] | str | None): Table names to find and read data from. If not provided
                or None, matches the closest names available in the database.
            *args: Additional positional arguments for extensibility and future compatibility.
            **kwargs: Additional keyword arguments for extensibility and future compatibility.

        Yields:
            ConnectorTable: A table object fetched from the database matching the specified or closest table
                names.
        """
        tbl_name2tbls = sqlite_connector.load_tables_from_database()
        tbl_in_db_to_analyze = utils_find_closest_matches(tbl_in_db_to_analyze, list(tbl_name2tbls.keys()))
        for tbl_name in tbl_in_db_to_analyze:
            yield tbl_name2tbls[tbl_name]

    def get_columns_no_pk_fk(self,
                             table: ConnectorTable,
                             start_from_cols: list[str] | None = None
                             ) -> list[str]:
        """
        Retrieves a list of column names excluding primary key, foreign key, and certain unwanted patterns.

        The function processes the provided or default column names of a table and excludes
        those that are primary keys, foreign keys, or contain specific substrings such as
        `id`, `code`, or `key` in their name.

        Args:
            table (ConnectorTable): The table containing metadata about the columns, as well as
                primary and foreign key information.
            start_from_cols (list[str] | None): Optional list of initial column names to process.
                If None, all column names from the table are used.

        Returns:
            list[str]: A filtered list of column names, excluding primary keys, foreign keys,
            and those containing `id`, `code`, or `key` substrings.
        """
        column_names = start_from_cols or list(table.tbl_col2metadata.keys())
        primary_keys_name = [pk.column_name for pk in table.primary_key] if table.primary_key else []
        primary_keys_name += [fk['parent_column'] for fk in table.foreign_keys] if table.foreign_keys else []
        column_names = [val for val in column_names if
                        val not in primary_keys_name and
                        'id' not in val.lower() and
                        'code' not in val.lower() and
                        'key' not in val.lower()]
        return column_names
