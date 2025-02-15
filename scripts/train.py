import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import dr_gen.utils.train_eval as te

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
    logging.info(">> Running Training")

    # Setup
    cfg.device = torch.device(cfg.device)
    generator = ru.set_deterministic(cfg.seed)

    # Data
    logging.info(" :: Loading Dataloaders :: ")
    split_dls = du.get_dataloaders(cfg, generator)
    logging.info(f">> Downloaded to: {cfg.paths.dataset_cache_root}")

    # Model
    model = create_model(cfg, len(split_dls['train'].dataset.classes))

    # Run Train
    train_loop(cfg, model, split_dls['train'], val_dl=split_dls['val'])



if __name__ == "__main__":
    run()
