import hydra
from omegaconf import DictConfig, OmegaConf

from dr_util.config_verification import validate_cfg
from dr_gen.utils.train_eval import GenMetrics
from dr_util.schemas import get_schema

from dr_gen.utils import data

import torch
import dr_gen.run_utils as ru
import dr_gen.data_utils as du
import dr_gen.model_utils as mu
import dr_gen.train_eval as te


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg: DictConfig):
    if not validate_cfg(cfg, "uses_metrics", get_schema):
        return

    # Make Metrics and Log Cfg
    md  = GenMetrics(cfg)
    md.log(cfg)
    cfg.md = md

    cfg.md.log(">> Running Training")

    # Setup
    cfg.device = torch.device(cfg.device)
    generator = ru.set_deterministic(cfg.seed)

    # Data
    cfg.md.log(" :: Loading Dataloaders :: ")
    split_dls = du.get_dataloaders(cfg, generator)
    cfg.md.log(f">> Downloaded to: {cfg.paths.dataset_cache_root}")

    # Model
    model = mu.create_model(cfg, len(split_dls["train"].dataset.classes))

    # Run Train
    #te.train_loop(cfg, model, split_dls["train"], val_dl=split_dls["val"])


if __name__ == "__main__":
    run()
