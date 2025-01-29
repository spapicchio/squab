import random
from collections import defaultdict
from typing import TypeAlias, Generator

from langchain_community.callbacks import get_openai_callback
from qatch.connectors import ConnectorTable, SqliteConnector

from ...utils import utils_get_db_dump_no_insert
from .... import DatasetGenerator
from ....models import create_default_gpt4o

PatternType: TypeAlias = dict[str, list[str] | float]
MetadataType: TypeAlias = dict[str, list[str] | float]
TestType: TypeAlias = dict[str, str | float]


def get_random_name_column(categorical_columns):
    filtered_columns = [col for col in categorical_columns
                        if 'name' in col.lower() and 'unnamed' not in col.lower()]
    return random.choice(filtered_columns) if filtered_columns else None


def _find_overlapping_column_values(
        table_name: str,
        column1: str,
        column2: str,
        sqlite_connector: SqliteConnector,
) -> dict[tuple, list]:
    LIMIT = 50  # Max number of rows to fetch

    # Extracted function to fetch data through query
    def fetch_data(tbl_name: str, col1: str, col2: str) -> list[list]:
        query = f'SELECT `{col1}`, `{col2}` FROM `{tbl_name}` LIMIT {LIMIT};'
        return sqlite_connector.run_query(query)

    # Extracted function to build dictionary of column relationships
    def create_column_values_associations(data: list[list]) -> defaultdict:
        # get for each col1_value, the set of col2_values
        value_col12values_col2 = defaultdict(set)
        for col1_value, col2_value in data:
            value_col12values_col2[col1_value].add(col2_value)
        return value_col12values_col2

    # Extracted function to find intersections between column values
    def find_intersections_among_col1_values(value_col12values_col2: defaultdict) -> dict[tuple, list]:
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

                intersection_values_col2 = value_col12values_col2[col1_value1].intersection(
                    value_col12values_col2[col1_value2]
                )
                intersection_values_col2 = [val for val in intersection_values_col2 if "'" not in str(val)]
                if intersection_values_col2:
                    col1_val1_val2_to_values_col2[(col1_value1, col1_value2)] = list(intersection_values_col2)
        return col1_val1_val2_to_values_col2

    # Main function logic
    data = fetch_data(table_name, column1, column2)
    column_relations = create_column_values_associations(data)
    overlapping_pairs = find_intersections_among_col1_values(column_relations)

    return overlapping_pairs


class AttachmentGenerator(DatasetGenerator[PatternType, MetadataType, TestType]):
    def __init__(self, seed=2023):
        super().__init__(seed)
        self.model_generation = create_default_gpt4o(hub_prompt='question_variability',
                                                     model_kwargs={'temperature': 0.5})

    @property
    def ambiguity_definition(self):
        return """
            Attachment ambiguity: Attachment ambiguity refers to situations where two phrases
            are connected with relative pronouns, and it is ambiguous if the second phrase is 
            attached to the end of the first phrase or the entire first phrase. 
            The ambiguity rise when there is a many-to-many relationship between two columns that have a
            'Entity' - 'Component' semantic relation and distinct values in the Entity columns have same value in
            the Component column. In the question, it is ambiguous whether the value in the component column
            has to be attached to only one of the value in the Entity columns or to both. 
            If possible, try to formulate the question without mentioning the column to project.
            """

    @property
    def ambiguity_examples(self):
        return """
            ### Example 1:
            queries: [
                "SELECT EventSpaces.Name \r\nFROM EventSpaces\r\nWHERE (EventSpaces.Event_Space = \"Banquet Hall\" OR EventSpaces.Event_Space = \"Conference Room\") AND EventSpaces.Capacity = 200",
                "SELECT EventSpaces.Name \r\nFROM EventSpaces\r\nWHERE EventSpaces.Event_Space = \"Banquet Hall\" OR EventSpaces.Event_Space = \"Conference Room\" AND EventSpaces.Capacity = 200"
            ]
            Entity: "Event_Space"
            Component: "Capacity"
            question: "List all banquet halls and conference rooms with a 200 person capacity."

            ### Example 2:
            queries: [
            "SELECT MusicPerformer.Name \r\nFROM MusicPerformer\r\nWHERE (MusicPerformer.MusicPerformerType = \"Jazz Musician\" OR MusicPerformer.MusicPerformerType = \"Rock Guitarist\") AND MusicPerformer.YearsInIndustry = 10",
            "SELECT MusicPerformer.Name \r\nFROM MusicPerformer\r\nWHERE MusicPerformer.MusicPerformerType = \"Jazz Musician\" OR MusicPerformer.MusicPerformerType = \"Rock Guitarist\" AND MusicPerformer.YearsInIndustry = 10"
            ]
            Entity: "MusicPerformerType"
            Component: "YearsInIndustry"
            question: "Display jazz musicians and rock guitarists who have been in the industry for 10 years."    
            """

    def pattern_identification(self, table: ConnectorTable, *args, **kwargs) -> Generator[PatternType, None, None]:
        # Get all non-PK and non-FK columns
        columns = self.get_columns_no_pk_fk(table)
        if len(columns) < 3:
            return

        # Get categorical columns and choose a column to project
        categorical_columns = list(table.cat_col2metadata.keys())
        column_to_project = get_random_name_column(categorical_columns)
        if not column_to_project:
            return
        # Filter columns not related to the projected column
        non_name_columns = [col for col in columns if col != column_to_project]
        categorical_columns.remove(column_to_project)

        # Iterate through categorical columns and perform matching
        while categorical_columns:
            entity_column = categorical_columns.pop()
            for component_column in non_name_columns:
                if component_column == entity_column:
                    continue

                # Check for overlapping columns
                col1_val1_val2_to_values_col2 = _find_overlapping_column_values(
                    table.tbl_name, entity_column, component_column, kwargs['sqlite_connector']
                )
                # sample only two overlapping groups in column_1 to avoid explosion
                sampled_entity_values = random.sample(
                    list(col1_val1_val2_to_values_col2.keys()),
                    min(2, len(col1_val1_val2_to_values_col2))
                )

                # Yield results for pattern identification
                for entity_value in sampled_entity_values:
                    yield {
                        'entity': entity_column,
                        'component': component_column,
                        'column_to_project': column_to_project,
                        'entity_values': list(entity_value),
                        'component_value': random.choice(col1_val1_val2_to_values_col2[entity_value]),
                    }

    def metadata_generator(self, pattern: PatternType, *args, **kwargs) -> Generator[MetadataType, None, None]:
        # the metadata is extracted with an Heuristics during pattern identification
        yield pattern

    def _build_sql_interpretations(self, metadata, tbl_name):
        component = metadata['component']
        entity = metadata['entity']
        column_project = metadata['column_to_project']

        value_group_1, value_group_2 = metadata['entity_values']
        value_intersection = metadata['component_value']

        condition_1 = (f"(`{entity}` = '{value_group_1}' OR `{entity}` = '{value_group_2}')"
                       f" AND `{component}` = '{value_intersection}'")
        condition_2 = (f"`{entity}` = '{value_group_1}' "
                       f"OR `{entity}` = '{value_group_2}' AND `{component}` = '{value_intersection}'")
        queries = [
            f'SELECT `{column_project}` FROM `{tbl_name}` WHERE {condition_1}',
            f'SELECT `{column_project}` FROM `{tbl_name}` WHERE {condition_2}',
        ]

        return queries

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
        yield {'question': generation['question'],
               'answer': sql_interpretations,
               'question_cost': cb.total_cost}
