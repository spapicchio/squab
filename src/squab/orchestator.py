from datasets import DatasetDict
from distilabel.pipeline import Pipeline
from distilabel.models.llms import LiteLLM
from litellm.proxy.proxy_server import use_background_health_checks


from squab.data_connectors.sqlite_connector import LoadSqliteDatabase
from squab.pattern_identification.pi_many_to_many import PIManyToMany
from squab.relational_metadata.rm_entity_component import RMEntityComponent
from squab.test_generation.test_generation_scope import TestGeneratorScope


class Orchestrator:
    """
    The Orchestra class is responsible for managing the orchestration of tasks.
    It handles the execution of tasks in a specific order and manages dependencies
    between them.
    """

    def __init__(self):
        self.llm = LiteLLM(model="gpt-3.5-turbo")
        self.generator_name2classess = {
            "scope": [PIManyToMany, RMEntityComponent, TestGeneratorScope],
            "attachment": [],
        }

    def _init_pipeline_scope(
        self,
        db_path,
        db_id: str = None,
        max_identified_patterns_per_tbl=2,
        pattern_identification_class=None,
        relational_metadata_class=None,
        test_generation_class=None,
    ):
        with Pipeline() as pipeline:  #
            connector = LoadSqliteDatabase(
                name="connector",
                db_path=db_path,
                db_id=db_id,
                batch_size=1,
            )
            pi_step = pattern_identification_class(
                name="pattern_identification",
                max_identified_patterns_per_tbl=max_identified_patterns_per_tbl,
            )

            rm_step = relational_metadata_class(name="relational_metadata")

            test_generation_class = test_generation_class(name="test_generation")

            connector >> pi_step >> rm_step >> test_generation_class

        return pipeline

    def run(
        self,
        generators: list = None,
        db_path: str = "../test/test_db.sqlite",
        db_id: str = None,
        max_identified_patterns_per_tbl=2,
    ):
        generator2dataset = dict()
        for generator in generators:
            if generator not in self.generator_name2classess:
                raise ValueError(
                    f"Generator {generator} not found. Available generators: {self.generator_name2classess.keys()}"
                )

            pipeline: Pipeline = self._init_pipeline_scope(
                db_path=db_path,
                db_id=db_id,
                max_identified_patterns_per_tbl=max_identified_patterns_per_tbl,
                pattern_identification_class=self.generator_name2classess[generator][0],
                relational_metadata_class=self.generator_name2classess[generator][1],
                test_generation_class=self.generator_name2classess[generator][2],
            )
            distiset = pipeline.run(use_cache=False)
            print(distiset)
            dataset = distiset["default"]["train"]
            dataset = dataset.remove_columns(["table"])
            generator2dataset[generator] = dataset
        return DatasetDict(generator2dataset)


if __name__ == "__main__":
    orchestrator = Orchestrator()
    dataset = orchestrator.run(
        generators=["scope"],
        db_path="/workspaces/squab/test/test_db.sqlite",
    )

    print(dataset)
    print(dataset["scope"].to_pandas().head())
