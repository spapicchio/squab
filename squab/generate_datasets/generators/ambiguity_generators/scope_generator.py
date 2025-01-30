import logging
from typing import Generator, TypeAlias, Literal

import pandas as pd
from langchain_community.callbacks import get_openai_callback
from qatch.connectors import ConnectorTable

from ...utils import utils_syntactic_match, utils_get_db_dump_no_insert
from .... import DatasetGenerator
from ....models import create_default_gpt4o
from ....models.langchain_wrapper import getter_json_output_from_resoning

PatternType: TypeAlias = dict[str, list[str] | float]
MetadataType: TypeAlias = dict[str, str | float]
TestType: TypeAlias = dict[str, str | float]


def _is_many_to_many(table: ConnectorTable, col1: str, col2: str, sqlite_connector) -> bool:
    # check for multiple unique values in col2 for each unique value in col1
    query = (f'SELECT `{col1}`, `{col2}` '
             f'FROM `{table.tbl_name}` LIMIT 100;')

    df = pd.read_sql_query(query, sqlite_connector.engine)

    cond1 = any(df.groupby(col1)[col2].nunique() > 1)
    # check for multiple unique values in col1 for each unique value in col2
    cond2 = any(df.groupby(col2)[col1].nunique() > 1)

    return cond1 and cond2


class ScopeGenerator(DatasetGenerator):
    def __init__(self, seed=2023):
        super().__init__(seed)
        self.model_generation = create_default_gpt4o(hub_prompt='question_variability',
                                                     model_kwargs={'temperature': 0.5})

        self.model_metadata = create_default_gpt4o(hub_prompt='scope_pattern_semantic',
                                                   model_kwargs={'temperature': 0.2})

    @property
    def test_category(self):
        return 'scope'

    @property
    def test_type(self) -> Literal['ambig', 'unans']:
        return 'ambig'

    @property
    def ambiguity_definition(self):
        return """
          Scope ambiguity occurs when it is unclear how a modifier or phrase is attached to the rest of the sentence.
          The ambiguity rise when there is a many-to-many relationship between two columns that have a
           'Entity' - 'Component' semantic relation. 
          Therefore, it is unclear whether the question is asking for all the component present in all the entities 
          (collective interpretation) or for each entity separately (distributive interpretation). 
          Consider the NL question "What activities does each gym offer?" over a table with a many-to-many relationship
          between Gym (entity) and Activities (component). 
          Here, there are two interpretations of the question: in the collective interpretation,
          the quantifier is interpreted widely (i.e., “each gym” refers to all gyms in the database).
          Instead, in the distributive interpretation, 
          the quantifier is interpreted narrowly (i.e., “each gym” is considered separately).
          """

    @property
    def ambiguity_examples(self):
        return """
          ### Example 1:
          queries:
          Collective Interpretation: "SELECT ClassName FROM GymClasses GROUP BY ClassID, ClassName HAVING COUNT(DISTINCT GymID) = (SELECT COUNT(DISTINCT GymID) FROM GymClasses);",
          Distributive Interpretation: "SELECT DISTINCT GymName, ClassName FROM GymClasses"
          question: "What activities does each gym offer?"
          ### Example 2:
          queries:
          Collective Interpretation: "SELECT `Genre` FROM `Movies` GROUP BY `Genre` HAVING COUNT(DISTINCT `Budget`) = (SELECT COUNT(DISTINCT `Budget`) FROM `Movies`);",
          Distributive Interpretation: "SELECT DISTINCT `Genre`, `Budget` FROM `Movies` "
          question: "List movie genres associated with the budgets of each movie."
          """

    def pattern_identification(self, table: ConnectorTable, *args, **kwargs) -> Generator[PatternType, None, None]:
        # get categorical column that are not primary key or foreign keys in the table
        column_names = self.get_columns_no_pk_fk(table, start_from_cols=list(table.cat_col2metadata.keys()))
        for i in range(len(column_names) - 1):
            for j in range(i + 1, len(column_names)):
                col1, col2 = column_names[i], column_names[j]
                if _is_many_to_many(table, col1=col1, col2=col2, sqlite_connector=kwargs['sqlite_connector']):
                    yield {'columns_in_many_2_many': [col1, col2]}

    def metadata_generator(self, pattern: PatternType, *args, **kwargs) -> Generator[MetadataType, None, None]:
        many_to_many_columns = pattern['columns_in_many_2_many']
        entity_component_json = self.model_metadata.predict({'names': ','.join(many_to_many_columns)})
        entity_component_json = getter_json_output_from_resoning(entity_component_json)
        if 'entity' not in entity_component_json or 'component' not in entity_component_json:
            return
        elif entity_component_json['entity'] is None and entity_component_json['component'] is None:
            return

        # to avoid hallucination error, use a syntactic score to get the correct column
        try:
            similarity_score_col_1 = utils_syntactic_match(entity_component_json['entity'], many_to_many_columns[0])
            similarity_score_col_2 = utils_syntactic_match(entity_component_json['entity'], many_to_many_columns[1])
        except TypeError as e:
            logging.warning(f"Error in SCOPE syntactic match: {many_to_many_columns}, {many_to_many_columns}")
            return

        yield {
            'entity': many_to_many_columns[0] if similarity_score_col_1 > similarity_score_col_2 else
            many_to_many_columns[1],
            'component': many_to_many_columns[1] if similarity_score_col_1 > similarity_score_col_2 else
            many_to_many_columns[0],
        }

    def _build_sql_interpretations(self, metadata, tbl_name):
        component = metadata['component']
        entity = metadata['entity']

        query_1 = (
            f"SELECT `{component}` "
            f"FROM `{tbl_name}` "
            f"GROUP BY `{component}` "
            f"HAVING COUNT(DISTINCT `{entity}`) = (SELECT COUNT(DISTINCT `{entity}`) FROM `{tbl_name}`)"
        )
        query_2 = (
            f"SELECT DISTINCT `{component}`, `{entity}` "
            f"FROM `{tbl_name}` "
        )

        return [query_1, query_2]

    def tests_generator(self, metadata: MetadataType, *args, **kwargs) -> Generator[TestType, None, None]:
        sql_interpretations = self._build_sql_interpretations(metadata, kwargs['table'].tbl_name)

        with get_openai_callback() as cb:
            # step 1: Generate question
            generation = self.model_generation.predict({
                'ambig_definition': self.ambiguity_definition,
                'ambig_example': self.ambiguity_examples,
                'queries': sql_interpretations,
                'metadata': metadata,
                'database': utils_get_db_dump_no_insert(kwargs['sqlite_connector'].db_path)
            })
            generation = getter_json_output_from_resoning(generation)
        yield {'question': generation['question'],
               'answer': sql_interpretations,
               'question_cost': cb.total_cost}
