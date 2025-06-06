import hydra
from langgraph.func import entrypoint, task
from omegaconf import DictConfig
from pydantic import BaseModel

from src.squab.graph_states import Line
from src.squab.nodes import node_read_db_sqlite, Generators, process_dataset_with_generator


# from langgraph.checkpoint.sqlite import SqliteSaver


# https://langchain-ai.github.io/langgraph/concepts/functional_api/#execution
# https://langchain-ai.github.io/langgraph/tutorials/workflows/?h=parallel#orchestrator-worker
#
# conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
# memory = SqliteSaver(conn)

@task
def node_aggregate_results(datasets: list[list[Line]]) -> list[Line]:
    """
    Aggregate results from multiple datasets.
    """
    aggregated: list[Line] = []
    for dataset in datasets:
        aggregated.extend(dataset)
    return aggregated


class WorkerInput(BaseModel):
    db_path: str
    db_id: str | None = None
    only_these_tbl: str | list[str] | None = None
    generators: dict[str, dict]


@entrypoint()
def orchestrator(
        worker_input: WorkerInput,
) -> list[Line]:
    # preprocess the data
    worker_input = worker_input.model_dump()
    processed_data = node_read_db_sqlite(**worker_input).result()
    # get the generators function
    generators = list(worker_input["generators"].keys()) or [name.value for name in Generators]
    datasets = [process_dataset_with_generator(processed_data, gen_name, worker_input['generators'][gen_name])
                for gen_name in generators]
    # wait for all generators to finish and aggregate the results
    aggregated_dataset = node_aggregate_results(
        [dataset.result() for dataset in datasets]
    )
    return aggregated_dataset.result()


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    input_ = WorkerInput(**cfg)
    dataset = orchestrator.invoke(input_)
    print(dataset)


if __name__ == "__main__":
    main()
