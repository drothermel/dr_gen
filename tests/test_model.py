import pytest
import torch
from torch.nn import CrossEntropyLoss
import timm
from omegaconf import OmegaConf

import dr_gen.models as mu

# --------- Tests for create_optim and create_lrsched ---------

@pytest.fixture
def full_cfg():
    cfg = OmegaConf.create(
        {
            "epochs": 10,
            "optim": {
                "name": "sgd",
                "lr": 0.01,
                "lr_scheduler": "steplr",
                "warmup_epochs": 5,
                "warmup_start_lr": 0.0,
                "lr_min": 0.0,
                "cycle_limit": 1,
                "step_size": 30,
                "gamma": 0.9,
                "momentum": 0.9,
                "loss": "cross_entropy",
            },
            "model": {
                "name": "resnet18", "weights": None, "source": "torchvision"
            },
            "device": "cpu",
            "metrics": {"loggers": []},
            "weight_type": "random",
            "load_checkpoint": None,
            "write_checkpoint": None,
            "md": None,
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
def test_create_lrsched(lrsched_type, expected_class, full_cfg) -> None:
    full_cfg.optim.lr_scheduler = lrsched_type
    # Use a dummy optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lr_sched = mu.create_lrsched(full_cfg, optimizer)
    assert isinstance(lr_sched, expected_class)


# --------- Test for create_model ---------


def test_create_model(full_cfg) -> None:
    num_classes = 10
    model = mu.create_model(full_cfg, num_classes)
    # Check that a model is created and is a torch.nn.Module
    assert isinstance(model, torch.nn.Module)
    # Optionally, if the model has a fully connected layer named "fc",
    # verify its output features equal num_classes.
    if hasattr(model, "fc"):
        assert model.fc.out_features == num_classes


# --------- Test for create_optim_lrsched ---------


def test_create_optim_lrsched(full_cfg) -> None:
    full_cfg.optim.name = "sgd"
    # Create a dummy model
    model = mu.create_model(full_cfg, num_classes=10)
    optimizer, lr_scheduler = mu.create_optim_lrsched(full_cfg, model)
    assert isinstance(optimizer, torch.optim.SGD)
    # Verify that the optimizer has the given lr and momentum.
    for group in optimizer.param_groups:
        assert group["lr"] == full_cfg.optim.lr
        assert group["momentum"] == full_cfg.optim.momentum
    # Verify that a StepLR scheduler was created.
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR)


# --------- Test for get_model_optim_lrsched ---------


def test_get_model_optim_lrsched(full_cfg) -> None:
    full_cfg.optim.name = "adamw"
    full_cfg.optim.lr_scheduler = "steplr"
    model, optimizer, lr_scheduler = mu.get_model_optim_lrsched(
        full_cfg, num_classes=10,
    )
    assert isinstance(model, torch.nn.Module)
    assert next(model.parameters()).device.type == "cpu"
    # Check that the optimizer is of the correct type
    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR)

# --------- Test for get_criterion ---------


def test_get_criterion(full_cfg) -> None:
    full_cfg.optim.loss = "cross_entropy"
    criterion = mu.get_criterion(full_cfg)
    assert isinstance(criterion, CrossEntropyLoss)


# --------- Test for checkpoint_model ---------


def test_checkpoint_model(tmp_path, full_cfg) -> None:
    # Create dummy model, optimizer, and lr_scheduler.
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    # Create a temporary directory for checkpointing.
    chpt_dir = tmp_path / "chkpts"
    full_cfg.write_checkpoint = str(chpt_dir)
    full_cfg.md = None
    checkpoint_name = "test_checkpoint"
    mu.checkpoint_model(
        full_cfg, model, checkpoint_name, optim=optimizer, lrsched=lr_scheduler
    )
    # Verify that the checkpoint file exists and contains the required keys.
    chpt_path = chpt_dir / f"{checkpoint_name}.pt"
    assert chpt_path.exists()
    chpt_data = torch.load(chpt_path, map_location="cpu", weights_only=False)
    for key in ["model", "optimizer", "lr_scheduler"]:
        assert key in chpt_data


# --------- Test for checkpoint loading ---------


def test_load_checkpoint_in_get_model_optim_lrsched(tmp_path, full_cfg) -> None:
    # Create initial model
    model, optimizer, lr_scheduler = mu.get_model_optim_lrsched(full_cfg, num_classes=2)

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
    full_cfg.load_checkpoint = str(chpt_path)

    # Load model with checkpoint
    loaded_model, loaded_optimizer, loaded_lr_scheduler = mu.get_model_optim_lrsched(
        full_cfg, num_classes=2
    )

    # Verify model is loaded and optimizer/scheduler state is preserved
    assert isinstance(loaded_model, torch.nn.Module)
    assert loaded_optimizer is not None
    assert loaded_lr_scheduler is not None
    assert loaded_optimizer == optimizer.state_dict()


def test_checkpoint_model_no_write_dir(full_cfg) -> None:
    # Test that checkpoint_model returns early when no write_checkpoint is set
    model = torch.nn.Linear(10, 2)
    full_cfg.write_checkpoint = None
    mu.checkpoint_model(full_cfg, model, "test_checkpoint")

