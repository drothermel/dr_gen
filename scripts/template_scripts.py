import hydra
from omegaconf import DictConfig

from dr_util.config_verification import validate_cfg
from dr_util.schemas import get_schema

from dr_gen.utils.metrics import GenMetrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg: DictConfig):
    if not validate_cfg(cfg, "uses_metrics", get_schema):
        return

    # Make Metrics and Log Cfg
    md = GenMetrics(cfg)
    md.log(cfg)
    md.log(">> Welcome to your new script!")


if __name__ == "__main__":
    run()
