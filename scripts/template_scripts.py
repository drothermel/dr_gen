import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from dr_util.config_verification import validate_cfg
from dr_util.metrics import Metrics
from dr_util.schemas import get_schema

from dr_gen.utils import data


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig):
    if not validate_cfg(cfg, "uses_metrics", get_schema):
        return

    # Make Metrics and Log Cfg
    md  = Metrics(cfg)
    md.log(cfg)

    """
    log_cfg(cfg)
    logging.info(">> Welcome to your new script! here")

    logging.info(" :: Loading Train Dataset :: ")
    data.get_dataset(cfg.data, cfg.paths, train=True)
    data.get_dataset(cfg.data, cfg.paths, train=False)
    logging.info(f">> Downloaded to: {cfg.paths.dataset_cache_root}")
    """


if __name__ == "__main__":
    run()
