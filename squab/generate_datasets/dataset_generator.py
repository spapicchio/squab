from abc import ABC, abstractmethod
from itertools import islice
from typing import Generator
from typing import Optional, Union

import pandas as pd
from pydantic import BaseModel, Field
from qatch.connectors import ConnectorTable, SqliteConnector

from .utils import utils_find_closest_matches


class DatasetInput(BaseModel):
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
    Provides an abstract interface for generating datasets composed of tables, patterns, metadata, and tests.

    Designed for creating test datasets by combining patterns, metadata, and test cases extracted
    from tables. Subclasses must implement abstract methods for specific functionality, such as
    identifying patterns, generating metadata, and producing test cases.

    Attributes:
        None
    """

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
    def metadata_generator(self, pattern, *args, **kwargs) -> Generator[MetadataType, None, None]:
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
                             relative_sqlite_db_path: str,
                             tbl_in_db_to_analyze: list[str] | str | None = None,
                             db_name: str | None = None,
                             tables: dict[str, pd.DataFrame] | None = None,
                             table2primary_key: dict[str, str] | None = None,
                             *args, **kwargs) -> Generator[ConnectorTable, None, None]:
        """
        Iterates over tables in a SQLite database based on specified criteria and yields them as a generator.

        This method allows users to read tables from a SQLite database by specifying the desired
        tables to analyze or letting the function handle table selection. It returns a generator
        that yields tables. The function employs utility methods for matching table names and leverages
        a SQLite connector for data loading.

        Args:
            relative_sqlite_db_path (str): The relative path to the SQLite database file.
            tbl_in_db_to_analyze (list[str] | str | None, optional): The name(s) of the table(s) to
                analyze. If not provided, matches the tables automatically.
            db_name (str | None, optional): The name of the database. Defaults to the name extracted
                from the SQLite file path if not provided.
            tables (dict[str, pd.DataFrame] | None, optional): A dictionary of pre-loaded tables. If
                specified, these tables will be considered for analysis instead of loading them anew.
            table2primary_key (dict[str, str] | None, optional): A dictionary mapping table names to their
                corresponding primary keys. This is passed to the SQLite connector.
            *args: Optional positional arguments.
            **kwargs: Optional keyword arguments.

        Yields:
            ConnectorTable: The loaded table objects matching the specified criteria.

        """
        db_name or relative_sqlite_db_path.split('/')[-1].replace('.sqlite', '')
        sqlite_connector = SqliteConnector(relative_db_path=relative_sqlite_db_path,
                                           db_name=db_name,
                                           tables=tables,
                                           table2primary_key=table2primary_key)
        tbl_name2tbls = sqlite_connector.load_tables_from_database()
        tbl_in_db_to_analyze = utils_find_closest_matches(tbl_in_db_to_analyze, list(tbl_name2tbls.keys()))
        for tbl_name in tbl_in_db_to_analyze:
            yield tbl_name2tbls[tbl_name]
