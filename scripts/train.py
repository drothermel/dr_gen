import hydra
from omegaconf import DictConfig


from dr_util.config_verification import validate_cfg

from dr_gen.schemas import get_schema
from dr_gen.utils.metrics import GenMetrics
import dr_gen.utils.run as ru
import dr_gen.utils.data as du
import dr_gen.utils.train_eval as te


def validate_run_cfg(cfg):
    settings_to_validate = [
        "uses_metrics",
        "uses_data",
        "uses_model",
        "uses_optim",
        "performs_run",
    ]
    validate_all = [validate_cfg(cfg, vs, get_schema) for vs in settings_to_validate]
    return all(validate_all)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg: DictConfig):
    if not validate_run_cfg(cfg):
        return

    # Make Metrics and Log Cfg
    md = GenMetrics(cfg)
    md.log(cfg)

    md.log(">> Running Training")

    # Setup
    generator = ru.set_deterministic(cfg.seed)

    # Data
    md.log(" :: Loading Dataloaders :: ")
    split_dls = du.get_dataloaders(cfg, generator)
    md.log(f">> Downloaded to: {cfg.paths.dataset_cache_root}")

    # Run Train
    te.train_loop(
        cfg,
        split_dls["train"],
        val_dl=split_dls["val"],
        eval_dl=split_dls["eval"],
        md=md,
    )
    md.log(">> End Run")


if __name__ == "__main__":
    run()
