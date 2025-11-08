import hydra
import importlib
from omegaconf import DictConfig
import os
from datetime import datetime
from evaluation.config import Config


def get_model_class(model_name):
    module_path, class_name = model_name.rsplit(".", 1)
    if "grader" in model_name:
        module = importlib.import_module(f"evaluation.grader.{module_path}")
    else:
        module = importlib.import_module(f"evaluation.eval.{module_path}")
    model_class = getattr(module, class_name)
    return model_class


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run_pipeline(cfg: Config) -> None:
    # Generate a unique run identifier (timestamp)
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
    print("Unique run ID:", unique_id)

    # Update unique_id in both configs
    cfg.dataset.params.results.unique_id = unique_id
    cfg.grader.params.results.unique_id = unique_id

    # Print the resolved path for debugging
    print(f"Using model checkpoint at: {cfg.dataset.params.model.path}")

    # Run evaluation
    print("\n=== Starting Evaluation ===")
    eval_class = get_model_class(cfg.dataset.name)
    eval_ = eval_class(**cfg.dataset.params, **cfg.llava_config)
    eval_.eval_model()

    # Run grading
    print("\n=== Starting Grading ===")
    grader_class = get_model_class(cfg.grader.name)
    grader_ = grader_class(**cfg.grader.params, **cfg.llava_config)
    grader_.grade()


if __name__ == "__main__":
    run_pipeline()
