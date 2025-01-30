import os
import random
from typing import Generator, TypeAlias, Literal

from langchain_community.callbacks import get_openai_callback
from langchain_openai.embeddings import OpenAIEmbeddings
from qatch.connectors import ConnectorTable

from .utils import utils_combine_clusters, utils_get_top_k_index_similar_matrix
from ...utils import utils_run_qatch, utils_get_db_dump_no_insert
from .... import DatasetGenerator
from ....models import create_default_gpt4o
from ....models.langchain_wrapper import getter_json_output_from_resoning

# Define reusable type aliases at the top
PatternType: TypeAlias = dict[str, list[str] | float]
MetadataType: TypeAlias = dict[str, str | float]
TestType: TypeAlias = dict[str, str | float]


class ColumnAmbiguityGenerator(DatasetGenerator[PatternType, MetadataType, TestType]):
    """
    A specialized dataset generator that handles column ambiguity in natural language
    queries on databases.

    This class is designed to identify ambiguous queries involving multiple
    columns with similar semantics, generate metadata to address such ambiguities,
    and create test cases to evaluate the handling of such queries. It leverages
    GPT-based models for interpretation and embeddings for comparing column
    semantics.

    Attributes:
        model_generation (Any): The language model used for question generation
            to resolve query ambiguities.
        encoder (OpenAIEmbeddings): The embedding generator for comparing
            semantics of table columns.
        metadata_generator (Any): The model used for generating labels to
            synthesize metadata for ambiguous columns.

    Properties:
        ambiguity_definition (str): A descriptive definition of column ambiguity,
            including examples of its occurrence.
        ambiguity_examples (str): Practical examples showcasing ambiguous
            queries, associated metadata, and rephrased ambiguous questions.
    """

    def __init__(self, seed=2023):
        super().__init__(seed)
        self.model_generation = create_default_gpt4o(hub_prompt='question_variability',
                                                     model_kwargs={'temperature': 0.5})
        self.encoder = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.model_metadata = create_default_gpt4o(hub_prompt='label_columns_selector')

    @property
    def test_type(self) -> Literal['ambig', 'unans']:
        return 'ambig'

    @property
    def test_category(self):
        return 'column_ambiguity'

    @property
    def ambiguity_definition(self):
        return """
            Colum Ambiguity arises when a natural language query is insufficiently specific to  identify
            a particular column within a table. This ambiguity often occurs when multiple columns 
            share similar meaning and it is possible to associate these columns to a common label.
            As example, consider a table with two columns: `Name` and `Surname`.
             A query like "What are the information of Simone?" is ambiguous because
            it's uncertain whether the query refers to the Name or the Surname or to both columns. 
            Given the queries, the semantic similar columns and the label to use in the generation,
            generate an ambiguous question that uses the label rather than the columns with the same intent of each
            query. Note that you can use also synonyms of the label as long as they are not present in the table.
            """

    @property
    def ambiguity_examples(self):
        return """
            ### Example 1
            metadata = {'label': 'ratings', "columns": ["customer_review", "difficulty_level"]}
            queries: [
                "Select Reviews.Hikes, Reviews.customer_review From Reviews",
                "Select Reviews.Hikes, Reviews.difficulty_level From Reviews",
                "Select Reviews.Hikes, Reviews.customer_review, difficulty_level From Reviews"
            ]
            ambiguous question: "What hikes do we have and what are their ratings?",
            ### Example 2
            metadata = {'label': 'life expectancy', "columns": ["average_years_of_life", "gender_specific_life_expectancy"]}
            queries: [
              "SELECT average_years_of_life\r\nFROM LifeExpectancies\r\nORDER BY region_id\r\nLIMIT 1;",
              "SELECT gender_specific_life_expectancy\r\nFROM LifeExpectancies\r\nORDER BY region_id\r\nLIMIT 1;",
              "SELECT average_years_of_life, gender_specific_life_expectancy\r\nFROM LifeExpectancies\r\nORDER BY region_id\r\nLIMIT 1;"
            ]
            ambiguous question: "What is the life expectancy of the region with the lowest ID?"

            ### Example 3
            metadata = {"label": "branch", "columns": ["street_name", "neighborhood"]}
            queries: [
              "SELECT street_name FROM branches WHERE branch_manager = 'David Black'",
              "SELECT neighborhood FROM branches WHERE branch_manager = 'David Black'",
              "SELECT street_name, neighborhood FROM branches WHERE branch_manager = 'David Black'"
            ]
            ambiguous question: "Where is the branch run by David Black located?"

            ### Example 4
            metadata = {"label": "dividend", "columns": ["Dividend_Percentage", "Dividend_Value"]}
            queries: [
              "SELECT Dividend_Percentage FROM Dividends ORDER BY ExpectedPaymentDate",
              "SELECT Dividend_Value FROM Dividends ORDER BY ExpectedPaymentDate",
              "SELECT Dividend_Percentage, Dividend_Value FROM Dividends ORDER BY ExpectedPaymentDate"
            ]
            ambiguous question: "Show me the dividend yield ordered from the lowest expected payment date."
            """

    def pattern_identification(self, table: ConnectorTable, *args, **kwargs) -> Generator[PatternType, None, None]:
        columns = self.get_columns_no_pk_fk(table)
        columns = [table.tbl_col2metadata[val] for val in columns]

        # you need at least two columns to find a pattern
        if len(columns) < 2:
            return

        with get_openai_callback() as cb:
            similar_columns = self._get_similar_values(
                columns,
                threshold_similar_values=0.60
            )

        for column_pairs in similar_columns:
            if len(column_pairs) > 1:
                yield {'similar_cols': column_pairs, 'pattern_cost': cb.total_cost}

    def metadata_generator(self, pattern: PatternType, *args, **kwargs) -> Generator[MetadataType, None, None]:
        table: ConnectorTable = kwargs['table']
        tbl_schema = list(table.tbl_col2metadata.keys())

        similar_cols = pattern['similar_cols']

        with get_openai_callback() as cb:
            label = self.model_metadata.predict({
                'tbl_schema': tbl_schema,
                'cols': similar_cols
            })
            label = getter_json_output_from_resoning(label)

            if 'label' not in label or any(label['label'].lower() == col.lower() for col in tbl_schema):
                return

        yield {'hypernym': label['label'], 'metadata_cost': cb.total_cost}

    def tests_generator(self, metadata: MetadataType, *args, **kwargs) -> Generator[TestType, None, None]:
        similar_cols = kwargs['pattern']['similar_cols']
        # randomly select one col in similar cols
        selected_col = random.choice(similar_cols)

        list_queries_with_selected_col = utils_run_qatch(sqlite_connector=kwargs['sqlite_connector'],
                                                         selected_col=selected_col,
                                                         tbl_name=kwargs['table'].tbl_name)

        for test_category_query_question_dict in list_queries_with_selected_col:
            sql_interpretations = self._build_sql_interpretations(test_category_query_question_dict['query'],
                                                                  similar_cols,
                                                                  selected_col)
            with get_openai_callback() as cb:
                generation = self.model_generation.predict({
                    'ambig_definition': self.ambiguity_definition,
                    'ambig_example': self.ambiguity_examples,
                    'queries': sql_interpretations,
                    'metadata': metadata,
                    'database': utils_get_db_dump_no_insert(kwargs['sqlite_connector'].db_path),
                })
                generation = getter_json_output_from_resoning(generation)

            yield {'question': generation['question'],
                   'question_template': test_category_query_question_dict['question'].replace(selected_col,
                                                                                              metadata['hypernym']),
                   'answer': sql_interpretations,
                   'sql_tag': test_category_query_question_dict['test_category'],
                   'question_cost': cb.total_cost}

    def _get_similar_values(self,
                            values: list[str],
                            threshold_similar_values,
                            ) -> list[list[str]]:
        at_most_k = int(len(values) / 2)
        at_most_k = 2 if at_most_k < 2 else at_most_k
        parsed_columns = [f'{col}' for col in values]

        vals_embeddings = self.encoder.embed_documents(parsed_columns)

        top_k_indexes = utils_get_top_k_index_similar_matrix(vals_embeddings,
                                                             at_most_k=at_most_k,
                                                             threshold=threshold_similar_values)

        cluster_name2similar_cols = {
            f'cluster_{i}': [values[j] for j in indexes] + [values[i]]
            for i, indexes in enumerate(top_k_indexes)
            if indexes
        }
        # combine the values which are subsets of each other
        cluster_name2similar_cols = utils_combine_clusters(cluster_name2similar_cols)

        return [[col.column_name for col in similar_cols]
                for similar_cols in cluster_name2similar_cols.values()]

    def _build_sql_interpretations(self, query, similar_cols, col_in_query):
        sql_interpretations = []
        for col in similar_cols:
            sql_interpretations.append(query.replace(f'`{col_in_query}`', f"`{col}`"))

        # add all the similar cols only when are in projection and not in an aggregation
        before_from, after_from = query.lower().split('from')
        if (col_in_query in before_from) and (
                col_in_query not in after_from) and f'(`{col_in_query}`)' not in query.lower():
            sql_interpretations.append(
                query.replace(f'`{col_in_query}`', ", ".join(f"`{col}`" for col in similar_cols)))
        elif f'order by `{col_in_query}`' in after_from:
            sql_interpretations.append(
                query.replace(f'`{col_in_query}`', ", ".join(f"`{col}`" for col in similar_cols)))

        return sql_interpretations
