import os
from datasets import DatasetDict
from distilabel.pipeline import Pipeline
from distilabel.models.llms import LiteLLM

from squab.data_connectors.sqlite_connector import LoadSqliteDatabase
from squab.pattern_identification import (
    PIManyToMany,
    PISimilarColumns,
    PIOverlappingCols,
    PICategoryColumn,
)
from squab.relational_metadata import (
    RMEntityComponent,
    RMHypernym,
    RMOverlappingColValues,
    RMCategoryTokenName,
)
from squab.test_generation import (
    TestGeneratorScope,
    TestGeneratorColAmb,
    TestGeneratorAttachment,
    TestGeneratorTokenType,
)


class Orchestrator:
    """
    The Orchestrator class is responsible for managing the orchestration of tasks.
    It handles the execution of tasks in a specific order and manages dependencies
    between them.
    """

    def __init__(self):
        self.llm = LiteLLM(model="gpt-3.5-turbo")
        self.generator_name2classess = {
            "scope": [PIManyToMany, RMEntityComponent, TestGeneratorScope],
            "column_ambiguity": [PISimilarColumns, RMHypernym, TestGeneratorColAmb],
            "attachment": [
                PIOverlappingCols,
                RMOverlappingColValues,
                TestGeneratorAttachment,
            ],
            "token_type_ambiguity": [
                PICategoryColumn,
                RMCategoryTokenName,
                TestGeneratorTokenType,
            ],
        }

    def run(
        self,
        generators: list = None,
        db_path: str = "../test/test_db.sqlite",
        db_id: str = None,
        max_identified_patterns_per_tbl=2,
    ) -> DatasetDict:
        generator2dataset = dict()
        for generator in generators:
            if generator not in self.generator_name2classess:
                raise ValueError(
                    f"Generator {generator} not found. Available generators: {self.generator_name2classess.keys()}"
                )
            print(f"Running generator: {generator}")
            pipeline: Pipeline = self._init_pipeline_scope(
                db_path=db_path,
                db_id=db_id,
                max_identified_patterns_per_tbl=max_identified_patterns_per_tbl,
                pattern_identification_class=self.generator_name2classess[generator][0],
                relational_metadata_class=self.generator_name2classess[generator][1],
                test_generation_class=self.generator_name2classess[generator][2],
                generator_name=generator,
            )
            distiset = pipeline.run(use_cache=False)
            dataset = distiset["default"]["train"]
            dataset = dataset.remove_columns(["table"])
            dataset = dataset.filter(lambda x: x["test_question"])
            if len(dataset) == 0:
                print(f"Generator {generator} returned an empty dataset. Skipping it.")
                continue
            generator2dataset[generator] = dataset
        return DatasetDict(generator2dataset)


if __name__ == "__main__":
    orchestrator = Orchestrator()
    dataset = orchestrator.run(
        # generators=["scope", "attachment", "column_ambiguity"],
        # generators=["attachment"],
        generators=["token_type_ambiguity"],
        db_path="/workspaces/squab/test/test_db.sqlite",
    )
    # dataset.push_to_hub(
    #     "simone-papicchio/squab_test_generation",
    #     private=True,
    #     token=os.getenv("HUGGINGFACE_API_TOKEN"),
    # )

    for name, ds in dataset.items():
        ds.to_pandas().to_csv(f"{name}_test.csv", index=False)
