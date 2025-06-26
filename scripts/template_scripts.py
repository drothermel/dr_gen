import logging

import hydra
from dr_util.config_verification import validate_cfg
from dr_util.schemas import get_schema
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg: DictConfig) -> None:
    if not validate_cfg(cfg, "uses_metrics", get_schema):
        return

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Log configuration and welcome message
    logger.info("Configuration: %s", cfg)
    logger.info(">> Welcome to your new script!")


if __name__ == "__main__":
    run()
