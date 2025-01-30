import random
from typing import TypeAlias, Generator, Literal

import sqlalchemy
from langchain_community.callbacks import get_openai_callback
from qatch.connectors import ConnectorTable, SqliteConnector

from ...utils import utils_run_qatch, utils_get_db_dump_no_insert
from .... import DatasetGenerator
from ....models import create_default_gpt4o
from ....models.langchain_wrapper import getter_json_output_from_resoning

PatternType: TypeAlias = dict[str, list[str] | float]
MetadataType: TypeAlias = dict[str, str | float]
TestType: TypeAlias = dict[str, str | float]


def check_unanswerability_query(query: str, sqlite_connector: SqliteConnector):
    try:
        sqlite_connector.run_query(query)
    except sqlalchemy.exc.OperationalError as e:
        if 'no such column' in str(e):
            return True
    else:
        return False


class ColumnUnanswerableGenerator(DatasetGenerator):
    def __init__(self, seed=2023):
        super().__init__(seed)
        self.model_unans_col_generator = create_default_gpt4o(hub_prompt='unanswerable-column_generation',
                                                              model_kwargs={'temperature': 0.5})
        self.model_question_generator = create_default_gpt4o(hub_prompt='sql-to-text',
                                                             model_kwargs={'temperature': 0.5})

    @property
    def test_type(self) -> Literal['ambig', 'unans']:
        return 'unans'

    @property
    def test_category(self):
        return 'column_unanswerable'

    def pattern_identification(self, table: ConnectorTable, *args, **kwargs) -> Generator[PatternType, None, None]:
        yield {
            'tbl_schema': list(table.tbl_col2metadata.keys()),
            'cat_col': random.choice(list(table.cat_col2metadata.keys())) if table.cat_col2metadata else None,
            'num_col': random.choice(list(table.num_col2metadata.keys())) if table.num_col2metadata else None,
        }

    def metadata_generator(self, pattern: PatternType, *args, **kwargs) -> Generator[MetadataType, None, None]:
        cat_col = pattern.pop('cat_col')
        num_col = pattern.pop('num_col')

        if cat_col is None:
            num_to_generate = f'5 numerical data type'
        elif num_col is None:
            num_to_generate = f'5 categorical data type'
        else:
            # TODO make it programmable
            num_to_generate = f'5'

        with get_openai_callback() as cb:
            llm_new_cols = self.model_unans_col_generator.predict({
                'num_to_generate': num_to_generate,
                'db_name': kwargs['sqlite_connector'].db_id,
                'tbl_name': kwargs['table'].tbl_name,
                'tbl_schema': pattern['tbl_schema']
            })

        llm_new_cols = getter_json_output_from_resoning(llm_new_cols)
        if 'suggested_columns' not in llm_new_cols:
            return

        new_cols = [new_col for new_col in llm_new_cols['suggested_columns']
                    if 'column_name' in new_col and 'column_type' in new_col]

        for new_col in new_cols:
            if new_col['column_type'] == 'categorical' and cat_col is None:
                continue
            if new_col['column_type'] == 'numerical' and num_col is None:
                continue

            selected_col = cat_col if new_col['column_type'] == 'categorical' else num_col
            yield {'new_column_name': new_col['column_name'],
                   'new_column_type': new_col['column_type'],
                   'col_to_use_for_generation': selected_col}

    def tests_generator(self, metadata: MetadataType, *args, **kwargs) -> Generator[TestType, None, None]:
        col_to_use_for_generation = metadata['col_to_use_for_generation']
        list_queries_with_selected_col = utils_run_qatch(sqlite_connector=kwargs['sqlite_connector'],
                                                         selected_col=col_to_use_for_generation,
                                                         tbl_name=kwargs['table'].tbl_name)
        for test_category_query_question_dict in list_queries_with_selected_col:
            unans_query = test_category_query_question_dict['query'].replace(f'`{col_to_use_for_generation}`',
                                                                             f"`{metadata['new_column_name']}`")
            if check_unanswerability_query(unans_query, kwargs['sqlite_connector']):
                with get_openai_callback() as cb:
                    generated_question = self.model_question_generator.predict({
                        'examples': '',  # TODO add examples
                        'queries': unans_query,
                        'metadata': metadata,
                        'database': utils_get_db_dump_no_insert(kwargs['sqlite_connector']),
                    })
                generated_question = getter_json_output_from_resoning(generated_question)
                if 'question' not in generated_question:
                    continue

                question_template = test_category_query_question_dict['question'].replace(col_to_use_for_generation,
                                                                                          metadata['new_column_name'])
                yield {'question': generated_question['question'],
                       'question_template': question_template,
                       'query': unans_query,
                       'answer': 'UNANSWERABLE',
                       'sql_tag': test_category_query_question_dict['test_category'],
                       'question_cost': cb.total_cost}
