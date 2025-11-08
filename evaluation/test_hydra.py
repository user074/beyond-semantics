import hydra
import importlib
from pathlib import Path
from hydra import compose
from evaluation.config import Config


def get_model_class(model_name):
    module_path, class_name = model_name.rsplit(".", 1)
    module = importlib.import_module(f"evaluation.eval.{module_path}")
    model_class = getattr(module, class_name)
    return model_class


# CONF_DIR = Path(__file__).parent / "conf"  # or any absolute/relative path
@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: Config):
    print(f"Using model checkpoint at: {cfg.dataset.params.model.path}")
    eval_class = get_model_class(cfg.dataset.name)
    print(f"Retrieved evaluation class: {eval_class}")
    eval_ = eval_class(**cfg.dataset.params, **cfg.llava_config)
    eval_.eval_model()


if __name__ == "__main__":
    main()
