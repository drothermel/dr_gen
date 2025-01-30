import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# Potentially add all of this to dr_util
def cfg_to_loggable_lines(cfg):
    if isinstance(cfg, dict):
        cfg_str = str(cfg)
    else:
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg_str = OmegaConf.to_yaml(resolved_cfg)
    return cfg_str.strip("\n").split("\n")

def log_cfg(cfg):
    logging.info("="*19 + "   Config   " + "="*19)
    for cl in cfg_to_loggable_lines(cfg):
        logging.info(cl)
    logging.info("="*50)
    logging.info("")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig):
    log_cfg(cfg)
    logging.info(">> Welcome to your new script! here")
    
if __name__ == "__main__":
    run()

