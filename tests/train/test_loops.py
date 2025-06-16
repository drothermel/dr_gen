import pytest
import torch
import torch.nn as nn
from dr_util.metrics import BATCH_KEY
from omegaconf import OmegaConf

import dr_gen.train.evaluate as eu
import dr_gen.train.loops as te
import dr_gen.train.model as mu


# ----------------------
# Dummy Metrics Logger
# ----------------------
class DummyMetrics:
    def __init__(self):
        self.logs = []  # collect all log calls for inspection

    def log_data(self, data, group_name, ns=None):
        self.logs.append(("log_data", data))

    def log(self, msg):
        self.logs.append(("log", msg))

    def agg_log(self, group):
        self.logs.append(("agg_log", group))

    def clear_data(self):
        self.logs.append(("clear_data", None))


# ----------------------
# Dummy Functions & Dataloader
# ----------------------
def dummy_accuracy(output, target, topk=(1,)):
    # For testing, simply return fixed perfect accuracies.
    return torch.tensor([100.0, 100.0])


# Dummy implementations to override model utils functions.
def dummy_get_criterion(cfg):
    return nn.CrossEntropyLoss()


def dummy_checkpoint_model(cfg, model, name, optim=None, lrsched=None, md=None):
    # Do nothing for checkpointing during tests.
    pass


# A dummy dataset that returns a fixed tensor and target.
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=4):
        self.num_samples = num_samples
        self.classes = [0, 1, 2, 4, 5]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return a dummy "image" (e.g. 3x32x32) and a target label (0)
        return torch.randn(3, 32, 32), torch.tensor(0)


@pytest.fixture
def dummy_dataloader():
    dataset = DummyDataset()
    # Use a standard PyTorch DataLoader.
    return torch.utils.data.DataLoader(dataset, batch_size=2)


@pytest.fixture
def dummy_model():
    # Use a simple model (flatten + linear) for testing.
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))


@pytest.fixture
def dummy_cfg(tmp_path):
    # Create a dummy OmegaConf configuration.
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "epochs": 2,
            "train": {"run": True},
            "val": {"run": True},
            "eval": {"run": True},
            "optim": {
                "lr": 0.01,
                "name": "sgd",
                "loss": "cross_entropy",
            },
            "model": {
                "name": "resnet18",
                # Not used because we override model creation in tests.
                "weights": None,
            },
            "write_checkpoint": str(tmp_path / "checkpoints"),
        }
    )


# ----------------------
# Tests for log_metrics
# ----------------------
def test_log_metrics(dummy_cfg) -> None:
    md = DummyMetrics()
    # Monkey-patch the accuracy function used in log_metrics.
    original_accuracy = eu.accuracy
    eu.accuracy = dummy_accuracy
    try:
        # Create dummy tensors.
        output = torch.randn(4, 10)
        target = torch.zeros(4, dtype=torch.long)
        loss = torch.tensor(0.5)
        te.log_metrics(md, "train", loss=loss, output=output, target=target)

        logs = md.logs
        # Expect a log_data call with BATCH_KEY and output batch size.
        assert any(item == ("log_data", (BATCH_KEY, output.shape[0])) for item in logs)
        # Expect logs for acc1 and acc5 from our dummy_accuracy.
        assert any(item == ("log_data", ("acc1", 100.0)) for item in logs)
        assert any(item == ("log_data", ("acc5", 100.0)) for item in logs)
        # Expect a log_data call for loss.
        assert any(item == ("log_data", ("loss", loss)) for item in logs)
    finally:
        eu.accuracy = original_accuracy


# ----------------------
# Tests for train_epoch
# ----------------------
def test_train_epoch(dummy_cfg, dummy_dataloader, dummy_model) -> None:
    md = DummyMetrics()
    # Use a simple criterion and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(dummy_model.parameters(), lr=dummy_cfg.optim.lr)
    # Run one epoch of training.
    te.train_epoch(
        dummy_cfg,
        0,
        dummy_model,
        dummy_dataloader,
        criterion,
        optimizer,
        md=md,
    )
    # Check that some log entries were recorded.
    assert len(md.logs) > 0


# ----------------------
# Tests for eval_model
# ----------------------
def test_eval_model(dummy_cfg, dummy_dataloader, dummy_model) -> None:
    md = DummyMetrics()
    criterion = nn.CrossEntropyLoss()
    # Run evaluation and get back the metrics object.
    te.eval_model(dummy_cfg, dummy_model, dummy_dataloader, criterion, md=md)
    # Verify that some logging occurred.
    assert len(md.logs) > 0


# ----------------------
# Test for train_loop
# ----------------------
def test_train_loop(dummy_cfg, dummy_dataloader, dummy_model, monkeypatch) -> None:
    md = DummyMetrics()
    # Override model utility functions so that no real file I/O or
    # complex behavior occurs.
    monkeyatch_set = [
        (mu, "checkpoint_model", dummy_checkpoint_model),
    ]
    for module, name, func in monkeyatch_set:
        monkeypatch.setattr(module, name, func)

    # For this test, use the dummy dataloader for both train and validation.
    train_dl = dummy_dataloader
    val_dl = dummy_dataloader

    # Run the training loop.
    te.train_loop(dummy_cfg, train_dl, val_dl=val_dl, md=md)

    logs = md.logs
    # Verify that for each epoch, a start message was logged.
    start_epoch_logs = [
        msg for typ, msg in logs if typ == "log" and "Start Epoch" in msg
    ]
    assert len(start_epoch_logs) >= dummy_cfg.epochs

    # Verify that aggregation logs were recorded for "train" and "val".
    agg_logs = [entry for entry in logs if entry[0] == "agg_log"]
    groups_logged = {entry[1] for entry in agg_logs}
    assert "train" in groups_logged
    assert "val" in groups_logged

    # Finally, verify that a training time log was output.
    time_logs = [msg for typ, msg in logs if typ == "log" and "Training time" in msg]
    assert time_logs, "Expected a training time log at the end of train_loop"
