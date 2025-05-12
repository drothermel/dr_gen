import hydra
from omegaconf import DictConfig
from torch.utils.data import (
    RandomSampler,
)

import dr_util.determinism_utils as dtu
from dr_util.config_verification import validate_cfg

from dr_gen.schemas import get_schema
from dr_gen.utils.metrics import GenMetrics
from dr_gen.data.load_data import get_dataloaders_refactored
from dr_gen.train.loops import train_loop


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

    # Interpret weight initialization
    if cfg.weight_type == "pretrained":
        cfg.model.weights = "DEFAULT"
    elif cfg.weight_type == "scratch":
        cfg.model.weights = None
    else:
        assert False

    # Make Metrics and Log Cfg
    md = GenMetrics(cfg)
    md.log(cfg)

    md.log(">> Running Training")

    # Setup
    generator = dtu.set_deterministic(cfg.seed)

    # Data
    md.log(" :: Loading Dataloaders :: ")
    split_dls = get_dataloaders_refactored(cfg, generator)
    md.log(f">> Downloaded to: {cfg.paths.dataset_cache_root}")
    md.log("\n--- Dataloader Creation Summary ---")
    for name, loader in split_dls.items():
        md.log(f"DataLoader '{name}':")
        md.log(f"  Number of samples: {len(loader.dataset)}")
        md.log(f"  Batch size: {loader.batch_size}")
        md.log(f"  Shuffle: {isinstance(loader.sampler, RandomSampler)}")
        # Test iterating through one batch
        data_batch, label_batch = next(iter(loader))
        md.log(f"  First batch data shape: {data_batch.shape}, label shape: {label_batch.shape}")

    # Run Train
    train_loop(
        cfg,
        split_dls["train"],
        val_dl=split_dls["val"],
        eval_dl=split_dls["eval"],
        md=md,
    )
    md.log(">> End Run")


if __name__ == "__main__":
    run()
