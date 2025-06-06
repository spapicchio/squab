from typing import TypedDict

from langgraph.func import entrypoint, task

from src.squab.graph_states import Line
from src.squab.nodes import read_db_sqlite


# from langgraph.checkpoint.sqlite import SqliteSaver


# https://langchain-ai.github.io/langgraph/concepts/functional_api/#execution
# https://langchain-ai.github.io/langgraph/tutorials/workflows/?h=parallel#orchestrator-worker
#
# conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
# memory = SqliteSaver(conn)

@task
def aggregate_results(datasets: list[list[Line]]) -> list[Line]:
    """
    Aggregate results from multiple datasets.
    """
    aggregated: list[Line] = []
    for dataset in datasets:
        aggregated.extend(dataset)
    return aggregated


class WorkerInput(TypedDict):
    db_path: str
    db_id: str | None
    only_these_tbl: str | list[str] | None
    generators: list[str] | None


@entrypoint()
def orchestrator_worker(
        worker_input: WorkerInput,
) -> list[Line]:
    # preprocess the data
    processed_data = read_db_sqlite(**worker_input).result()
    # # get all the generators for the dataset
    # dataset_generators = get_generators_callable(generators)
    # # transform the data using the generators
    # datasets = [generator(processed_data) for generator in dataset_generators]
    # # wait for all generators to finish and aggregate the results
    # aggregated_dataset = aggregate_results(
    #     [dataset.result() for dataset in datasets]
    # ).result()
    # return aggregated_dataset
    return processed_data


if __name__ == "__main__":
    dataset = orchestrator_worker.invoke(
        {'db_path': 'fake_db_with_joins.sqlite',
         "only_these_tbl": "orders"}
    )

    print(dataset)
    print(len(dataset))
