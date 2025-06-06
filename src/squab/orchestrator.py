import hydra
from langgraph.func import entrypoint, task
from omegaconf import DictConfig
from pydantic import BaseModel

from squab.graph_states import Line
from squab.logger import get_logger
from squab.nodes import node_read_db_sqlite, Generators, process_dataset_with_generator


# https://langchain-ai.github.io/langgraph/concepts/functional_api/#execution
# https://langchain-ai.github.io/langgraph/tutorials/workflows/?h=parallel#orchestrator-worker

class WorkerInput(BaseModel):
    db_path: str
    db_id: str | None
    only_these_tbl: str | list[str] | None
    generators: dict[str, dict]

@task
def node_aggregate_results(datasets: list[list[Line]]) -> list[Line]:
    """
    Aggregate results from multiple datasets.
    """
    aggregated: list[Line] = []
    for dataset in datasets:
        aggregated.extend(dataset)
    return aggregated


@entrypoint()
def orchestrator(
        worker_input: WorkerInput,
) -> list[Line]:
    # preprocess the data
    worker_input = worker_input.model_dump()
    logger_orchestrator = get_logger('orchestrator')
    logger_orchestrator.info(f"Starting orchestrator for `db_path={worker_input['db_path']}`")
    processed_data = node_read_db_sqlite(**worker_input).result()
    logger_orchestrator.info(f"Finished reading `db_path={worker_input['db_path']}`")
    # get the generators function
    logger_orchestrator.info(f"Using generators: {worker_input['generators']}")
    generators = list(worker_input["generators"].keys()) or [name.value for name in Generators]
    datasets = [process_dataset_with_generator(processed_data, gen_name, worker_input['generators'][gen_name])
                for gen_name in generators]
    # wait for all generators to finish and aggregate the results
    aggregated_dataset = node_aggregate_results(
        [dataset.result() for dataset in datasets]
    )
    logger_orchestrator.info(f"Finished processing `db_path={worker_input['db_path']}`")
    return aggregated_dataset.result()


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    input_ = WorkerInput(**cfg)
    dataset = orchestrator.invoke(input_)
    print(dataset)


if __name__ == "__main__":
    main()
