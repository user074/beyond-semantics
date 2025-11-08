import hydra
import importlib
from evaluation.config import Config


def get_model_class(model_name):
    module_path, class_name = model_name.rsplit(".", 1)
    module = importlib.import_module(f"evaluation.eval.{module_path}")
    model_class = getattr(module, class_name)
    return model_class


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: Config):
    print(f"Using model checkpoint at: {cfg.dataset.params.model.path}")
    eval_class = get_model_class(cfg.dataset.name)
    eval_ = eval_class(**cfg.dataset.params, **cfg.llava_config)
    eval_.eval_model()



if __name__ == "__main__":
    main()
