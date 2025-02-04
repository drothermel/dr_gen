import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from dr_gen.utils import data


# Potentially add all of this to dr_util
def cfg_to_loggable_lines(cfg):
    if isinstance(cfg, dict):
        cfg_str = str(cfg)
    else:
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg_str = OmegaConf.to_yaml(resolved_cfg)
    return cfg_str.strip("\n").split("\n")


def log_cfg(cfg):
    logging.info("=" * 19 + "   Config   " + "=" * 19)
    for cl in cfg_to_loggable_lines(cfg):
        logging.info(cl)
    logging.info("=" * 50)
    logging.info("")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig):
    log_cfg(cfg)
    logging.info(">> Welcome to your new script! here")

    logging.info(" :: Loading Train Dataset :: ")
    train_data = data.get_dataset(cfg.data, cfg.paths, train=True)
    test_data = data.get_dataset(cfg.data, cfg.paths, train=False)
    logging.info(f">> Downloaded to: {cfg.paths.dataset_cache_root}")


if __name__ == "__main__":
    run()
