import random
import re
from typing import Generator, TypeAlias, Literal

import sqlalchemy
from langchain_community.callbacks import get_openai_callback
from qatch.connectors import ConnectorTable, SqliteConnector
from sqlalchemy import text

from ...utils import utils_run_qatch, utils_get_db_dump_no_insert
from .... import DatasetGenerator
from ....models import create_default_gpt4o
from ....models.langchain_wrapper import getter_json_output_from_resoning

PatternType: TypeAlias = dict[str, list[str] | float]
MetadataType: TypeAlias = dict[str, str | float]
TestType: TypeAlias = dict[str, str | float]

# Create a dictionary to serve as the local namespace for the exec function
local_namespace = {}


def check_unanswerability_query(query: str, udf_code: str, udf_name: str, sqlite_connector: SqliteConnector) -> bool:
    """
    Checks whether a given query is unanswerable in an SQLite database. If the query involves a UDF
    (User Defined Function) that doesn't exist, it attempts to load and use the provided UDF code.

    Args:
        query (str): The SQL query to be executed.
        udf_code (str): The code defining the User Defined Function.
        udf_name (str): Name of the User Defined Function.
        sqlite_connector (SqliteConnector): Object handling the SQLite connection.

    Returns:
        bool: True if the query is valid and answerable, False otherwise.
    """

    def execute_user_defined_function() -> bool:
        """
        Attempts to define and execute the user-defined function in the SQLite database.

        Returns:
            bool: True if the UDF is executed successfully, False otherwise.
        """
        try:
            # Dynamically execute the provided UDF code in a local namespace
            local_namespace = {}
            exec(udf_code, globals(), local_namespace)
        except Exception:
            return False

        # Extract the UDF callable and its argument count
        func_name = udf_name.split('(')[0]
        func = local_namespace.get(func_name)
        num_arguments = len(udf_name.split('(')[1].split(','))

        try:
            # Obtain a raw connection and create the UDF
            with sqlite_connector.engine.connect() as conn:
                conn.connection.create_function(func_name, num_arguments, func)
                # Execute the query to test the function
                conn.execute(text(query))
        except sqlalchemy.exc.OperationalError:
            return False

        return True

    try:
        # Attempt to execute the query directly
        sqlite_connector.run_query(query)
    except sqlalchemy.exc.OperationalError as e:
        # Check if the error is due to a missing function
        if 'no such function' in str(e):
            # Try to load and execute the UDF
            return execute_user_defined_function()

        return False
    # If the query executes without issues, it's not unanswerable
    return False


class CalculationUnanswerableGenerator(DatasetGenerator):
    def __init__(self, seed=2023):
        super().__init__(seed)
        self.model_unans_udf_generator = create_default_gpt4o(hub_prompt='unanswerable-udf_generation',
                                                              model_kwargs={'temperature': 0.5})

        self.model_question_generator = create_default_gpt4o(hub_prompt='sql-to-text',
                                                             model_kwargs={'temperature': 0.5})

    @property
    def test_type(self) -> Literal['ambig', 'unans']:
        return 'unans'

    @property
    def test_category(self):
        return 'calculation_unanswerable'

    def pattern_identification(self, table: ConnectorTable, *args, **kwargs) -> Generator[PatternType, None, None]:
        yield {
            'tbl_schema': list(table.tbl_col2metadata.keys()),
            'cat_col': random.choice(list(table.cat_col2metadata.keys())) if table.cat_col2metadata else None,
            'num_col': random.choice(list(table.num_col2metadata.keys())) if table.num_col2metadata else None,
        }

    def metadata_generator(self, pattern: PatternType, *args, **kwargs) -> Generator[MetadataType, None, None]:
        cat_col = pattern.pop('cat_col')
        num_col = pattern.pop('num_col')
        tbl_schema = pattern['tbl_schema']

        if cat_col is None:
            num_to_generate = f'2 UDFs with "udf_output_type" numerical'

        elif num_col is None:
            num_to_generate = f'2 UDFs with "udf_output_type" categorical'

        else:
            # TODO make it programmable
            num_to_generate = f'2 UDFs with "udf_output_type" mixed (categorical and numerical)'

        with get_openai_callback() as cb:
            llm_udf = self.model_unans_udf_generator.predict({
                'tbl_schema': tbl_schema,
                'num_to_generate': num_to_generate
            })
            udfs = llm_udf.split("# New UDF")
            for udf in udfs:
                # extract UDF
                udf_json = getter_json_output_from_resoning(udf)
                if 'udf_name' not in udf_json['generated_udf'] or 'udf_output_type' not in udf_json['generated_udf']:
                    continue
                # Extract Code
                python_matches = re.findall(r'```python.*?```', udf, re.DOTALL)
                if len(python_matches) == 0:
                    continue
                code = python_matches[-1].replace('```', '').replace('python', '')
                selected_col = cat_col if udf_json['udf_output_type'] == 'categorical' else num_col

                yield {'udf_name': udf_json['generated_udf']['udf_name'],
                       'udf_python_code': code,
                       'udf_output_type': udf_json['generated_udf']['udf_output_type'],
                       'udf_generation_cost': cb.total_cost / len(udfs),
                       'col_to_use_for_generation': selected_col}

    def tests_generator(self, metadata: MetadataType, *args, **kwargs) -> Generator[TestType, None, None]:
        col_to_use_for_generation = metadata['col_to_use_for_generation']
        list_queries_with_selected_col = utils_run_qatch(sqlite_connector=kwargs['sqlite_connectors'],
                                                         selected_col=col_to_use_for_generation,
                                                         tbl_name=kwargs['table'].tbl_name)

        for test_category_query_question_dict in list_queries_with_selected_col:
            unans_query = test_category_query_question_dict['query'].replace(f'`{col_to_use_for_generation}`',
                                                                             f"{metadata['udf_name']}")
            if check_unanswerability_query(unans_query, metadata['udf_python_code'], kwargs['sqlite_connectors']):
                with get_openai_callback() as cb:
                    generated_question = self.model_question_generator.predict({
                        'examples': '',  # TODO add examples
                        'queries': unans_query,
                        'metadata': metadata,
                        'database': utils_get_db_dump_no_insert(kwargs['sqlite_connectors']),
                    })
                generated_question = getter_json_output_from_resoning(generated_question)
                if 'question' not in generated_question:
                    continue
                udf_name = metadata['udf_name'].split('(')[0]
                question_template = test_category_query_question_dict['question'].replace(col_to_use_for_generation,
                                                                                          udf_name)
                yield {'question': generated_question['question'],
                       'question_template': question_template,
                       'answer': 'UNANSWERABLE',
                       'query': unans_query,
                       'sql_tag': test_category_query_question_dict['test_category'],
                       'question_cost': cb.total_cost}
