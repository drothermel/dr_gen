import pytest
import torch
import timm
from omegaconf import OmegaConf

import dr_gen.models as mu

# --------- Tests for create_optim and create_lrsched ---------

@pytest.fixture
def optim_cfg():
    lrsched_params = {**mu.LRSCHED_DEFAULTS}
    # If we need to supply extra params (e.g. step_size) we can add them:
    cfg = OmegaConf.create(
        {
            "epochs": 10,
            "lr": 0.01,
            "optim": {
                "lr_scheduler": "steplr",
                "warmup_epochs": 5,
                "warmup_start_lr": 0.0,
                "lr_min": 0.0,
                "cycle_limit": 1,
                "step_size": 30,
                "gamma": 0.9,
            },
        }
    )
    return cfg


@pytest.mark.parametrize(
    ("optim_type", "expected_class"),
    [
        ("sgd", torch.optim.SGD),
        ("rmsprop", torch.optim.RMSprop),
        ("adamw", torch.optim.AdamW),
    ],
)
def test_create_optim(optim_type, expected_class) -> None:
    # Create a dummy model with one parameter
    model = torch.nn.Linear(10, 1)
    model_params = model.parameters()
    # Build an optim_params dict including the required lr
    optim_params = {**mu.OPTIM_DEFAULTS, "lr": 0.01}
    optim = mu.create_optim(optim_type, model_params, optim_params)
    assert isinstance(optim, expected_class)


@pytest.mark.parametrize(
    ("lrsched_type", "expected_class"),
    [
        (None, type(None)),
        ("steplr", torch.optim.lr_scheduler.StepLR),
        ("timm_cosine", timm.scheduler.cosine_lr.CosineLRScheduler),
    ],
)
def test_create_lrsched(lrsched_type, expected_class, optim_cfg) -> None:
    optim_cfg.optim.lr_scheduler = lrsched_type
    # Use a dummy optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lr_sched = mu.create_lrsched(optim_cfg, optimizer)
    assert isinstance(lr_sched, expected_class)


# --------- Test for create_model ---------


def test_create_model() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"name": "resnet18", "weights": None},
        }
    )
    num_classes = 10
    model = mu.create_model(cfg, num_classes)
    # Check that a model is created and is a torch.nn.Module
    assert isinstance(model, torch.nn.Module)
    # Optionally, if the model has a fully connected layer named "fc",
    # verify its output features equal num_classes.
    if hasattr(model, "fc"):
        assert model.fc.out_features == num_classes


# --------- Test for create_optim_lrsched ---------


def test_create_optim_lrsched() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"name": "resnet18", "weights": None},
            "optim": {
                "name": "sgd",
                "lr": 0.05,
                "momentum": 0.9,
                "lr_scheduler": "steplr",
                "step_size": 3,
            },
            "device": "cpu",
            "metrics": {"loggers": []},  # for GenMetrics in other parts
        }
    )
    # Create a dummy model
    model = mu.create_model(cfg, num_classes=10)
    optimizer, lr_scheduler = mu.create_optim_lrsched(cfg, model)
    assert isinstance(optimizer, torch.optim.SGD)
    # Verify that the optimizer has the given lr and momentum.
    for group in optimizer.param_groups:
        assert group["lr"] == 0.05
        assert group["momentum"] == 0.9
    # Verify that a StepLR scheduler was created.
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR)


# --------- Test for get_model_optim_lrsched ---------


def test_get_model_optim_lrsched() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"name": "resnet18", "weights": None},
            "optim": {
                "name": "adamw",
                "lr": 0.001,
                # no lr_scheduler provided so create_optim_lrsched will be used
            },
            "device": "cpu",
            "metrics": {"loggers": []},
        }
    )
    model, optimizer, lr_scheduler = mu.get_model_optim_lrsched(cfg, num_classes=10)
    assert isinstance(model, torch.nn.Module)
    assert next(model.parameters()).device.type == "cpu"
    # Check that the optimizer is of the correct type
    assert isinstance(optimizer, torch.optim.AdamW)
    # If no lr_scheduler was provided in the config, one will be created;
    # it might be None if create_lrsched returns None.
    # We simply check that lr_scheduler is either None or an instance of a scheduler.
    if lr_scheduler is not None:
        assert isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler)  # noqa: SLF001


# --------- Test for get_criterion ---------


def test_get_criterion() -> None:
    cfg = OmegaConf.create(
        {
            "optim": {
                "loss": "cross_entropy",
                # Using default CRITERION_DEFAULTS so no extra params provided
            }
        }
    )
    criterion = mu.get_criterion(cfg)
    from torch.nn import CrossEntropyLoss

    assert isinstance(criterion, CrossEntropyLoss)


# --------- Test for checkpoint_model ---------


def test_checkpoint_model(tmp_path) -> None:
    # Create dummy model, optimizer, and lr_scheduler.
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    # Create a temporary directory for checkpointing.
    chpt_dir = tmp_path / "chkpts"
    cfg = OmegaConf.create(
        {
            "write_checkpoint": str(chpt_dir),
            "md": None,  # no logger needed
        }
    )
    checkpoint_name = "test_checkpoint"
    mu.checkpoint_model(
        cfg, model, checkpoint_name, optim=optimizer, lrsched=lr_scheduler
    )
    # Verify that the checkpoint file exists and contains the required keys.
    chpt_path = chpt_dir / f"{checkpoint_name}.pt"
    assert chpt_path.exists()
    chpt_data = torch.load(chpt_path, map_location="cpu", weights_only=False)
    for key in ["model", "optimizer", "lr_scheduler"]:
        assert key in chpt_data


# --------- Test for checkpoint loading ---------


def test_load_checkpoint_in_get_model_optim_lrsched(tmp_path) -> None:
    # First create the same model architecture we'll load into
    cfg = OmegaConf.create(
        {
            "model": {"name": "resnet18", "weights": None, "source": "timm"},
            "optim": {"name": "sgd", "lr": 0.05, "momentum": 0.9},
            "device": "cpu",
            "metrics": {"loggers": []},
            "weight_type": "random",
        }
    )

    # Create initial model
    model, optimizer, lr_scheduler = mu.get_model_optim_lrsched(cfg, num_classes=2)

    # Step the scheduler to change its state
    lr_scheduler.step()

    # Save checkpoint
    chpt_path = tmp_path / "test_checkpoint.pt"
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    torch.save(checkpoint, chpt_path)

    # Create new config that loads the checkpoint
    cfg_load = OmegaConf.create(
        {
            "model": {"name": "resnet18", "weights": None, "source": "timm"},
            "optim": {"name": "sgd", "lr": 0.01, "momentum": 0.8},
            "device": "cpu",
            "load_checkpoint": str(chpt_path),
            "metrics": {"loggers": []},
            "weight_type": "random",
        }
    )

    # Load model with checkpoint
    loaded_model, loaded_optimizer, loaded_lr_scheduler = mu.get_model_optim_lrsched(
        cfg_load, num_classes=2
    )

    # Verify model is loaded and optimizer/scheduler state is preserved
    assert isinstance(loaded_model, torch.nn.Module)
    assert loaded_optimizer is not None
    assert loaded_lr_scheduler is not None
    # The loaded optimizer should have the state from the checkpoint
    assert len(loaded_optimizer.state_dict()["state"]) > 0


def test_checkpoint_model_no_write_dir() -> None:
    # Test that checkpoint_model returns early when no write_checkpoint is set
    model = torch.nn.Linear(10, 2)
    cfg = OmegaConf.create({})  # No write_checkpoint

    # Should return without error
    mu.checkpoint_model(cfg, model, "test_checkpoint")

    # Also test with explicit None
    cfg = OmegaConf.create({"write_checkpoint": None})
    mu.checkpoint_model(cfg, model, "test_checkpoint2")
