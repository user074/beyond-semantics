import hydra
import importlib
from evaluation.config import Config


def get_model_class(model_name):
    module_path, class_name = model_name.rsplit(".", 1)
    module = importlib.import_module(f"evaluation.grader.{module_path}")
    model_class = getattr(module, class_name)
    return model_class


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: Config):
    grader_class = get_model_class(cfg.grader.name)
    grader_ = grader_class(**cfg.grader.params, **cfg.llava_config)
    grader_.grade()


if __name__ == "__main__":
    main()
