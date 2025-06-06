import hydra
from omegaconf import DictConfig

from squab.orchestrator import WorkerInput, orchestrator


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    input_ = WorkerInput(**cfg)
    dataset = orchestrator.invoke(input_)
    print(dataset)


if __name__ == "__main__":
    main()
