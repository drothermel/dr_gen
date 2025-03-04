from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np


def get_plt_cfg(**kwargs):
    base_plt_cfg = OmegaConf.create(
        {
            "figsize": (10, 6),
            "legend": False,
            "grid": True,
            "nbins": 10,
            "hist_range": None,
            "title": None,
            "xlabel": None,
            "ylabel": None,
            "labels": None,
            "xlim": None,
            "ylim": None,
            "density": False,
        }
    )
    plc = base_plt_cfg.copy()
    plc.hist_range = (80.0, 100.0)
    plc.nbins = 100
    plc.legend = True

    for k, v in kwargs.items():
        if k in plc:
            plc[k] = v
    return plc


def plcv(plc, key, default):
    if key in plc and plc[key] is not None:
        return plc[key]
    return default


def plot_lines(
    plc,
    data_lists,
):
    if len(data_lists) == 0 or len(data_lists[0]) == 0:
        print(">> invalid data")
        return
    plt.figure(figsize=plc.figsize)
    labels = plcv(plc, "labels", range(len(data_lists)))
    x = range(len(data_lists[0]))
    for i, data in enumerate(data_lists):
        plt.plot(x, data, linestyle="-", label=labels[i])

    plt.xlabel(plcv(plc, "xlabel", "Epoch"))
    plt.ylabel(plcv(plc, "ylabel", "Loss"))
    plt.title(plcv(plc, "title", "Loss During Training"))
    plt.grid(plc.grid)
    if plc.legend:
        plt.legend()
    if plc.xlim is not None:
        plt.xlim(plc.xlim)
    if plc.ylim is not None:
        plt.ylim(plc.ylim)
    plt.show()


def get_runs_data_stats(runs_data):
    n = len(runs_data)
    v_array = np.array(runs_data)
    v_mean = np.mean(v_array, axis=0)
    v_std = np.std(v_array, axis=0)
    v_sem = v_std / np.sqrt(n)
    return {
        "n": n,
        "epochs": np.array(range(len(runs_data[0]))),
        "mean": v_mean,
        "std": v_std,
        "sem": v_sem,
        "std_lower": v_mean - v_std,
        "std_upper": v_mean + v_std,
        "sem_lower": v_mean - v_sem,
        "sem_upper": v_mean + v_sem,
        "min": np.min(v_array, axis=0),
        "max": np.max(v_array, axis=0),
    }


def get_runs_data_stats_ind(runs_data, ind):
    stats = get_runs_data_stats(runs_data)
    ind_stats = {k: v[ind] for k, v in stats.items() if k != "n"}
    ind_stats["n"] = stats["n"]
    ind_stats["vals"] = [rd[ind] for rd in runs_data]
    return ind_stats


# Expected: [list_of_data = [metrics_per_epoch ...] ...]
def plot_summary_lines(
    plc,
    data_lists,
):
    # Expected data_list shapes: Num Datasets x Num Runs x Num Epochs
    #    but Num Runs can vary between datasets
    assert len(data_lists) > 0, ">> Invalid Data: Need at least one dataset"
    assert len(data_lists[0]) > 0, ">> Inv. Data: Dataset needs at least one run"
    assert len(data_lists[0][0]) > 0, ">> Inv. D: Run needs at least one epoch metric"

    plt.figure(figsize=plc.figsize)
    labels = plcv(plc, "labels", ["train mean", "val mean", "eval mean"])

    for i, runs_data in enumerate(data_lists):
        rd_stats = get_runs_data_stats(runs_data)
        x = rd_stats["epochs"]
        (line_mean,) = plt.plot(x, rd_stats["mean"], linewidth=3, label=labels[i])
        color = line_mean.get_color()
        plt.fill_between(x, rd_stats["min"], rd_stats["max"], color=color, alpha=0.1)
        plt.fill_between(
            x, rd_stats["std_lower"], rd_stats["std_upper"], color=color, alpha=0.3
        )
        plt.plot(
            x,
            rd_stats["sem_lower"],
            linestyle="-",
            linewidth=1,
            color=color,
            alpha=0.5,
        )
        plt.plot(
            x,
            rd_stats["sem_upper"],
            linestyle="-",
            linewidth=1,
            color=color,
            alpha=0.5,
        )

    plt.xlabel(plcv(plc, "xlabel", "Epoch"))
    plt.ylabel(plcv(plc, "ylabel", "Loss"))
    plt.title(plcv(plc, "title", "Loss During Training"))
    plt.grid(plc.grid)
    if plc.legend:
        plt.legend()
    if plc.xlim is not None:
        plt.xlim(plc.xlim)
    if plc.ylim is not None:
        plt.ylim(plc.ylim)
    plt.show()


def plot_cdf(plc, vals, cdf):
    plt.figure(figsize=plc.figsize)
    # labels = plcv(plc, 'labels', range(len(data_lists)))
    plt.plot(vals, cdf, linestyle="-", label="CDF")
    plt.fill_between(vals, cdf, color="skyblue", alpha=0.4)

    plt.xlabel(plcv(plc, "xlabel", "Epoch"))
    plt.ylabel(plcv(plc, "ylabel", "Loss"))
    plt.title(plcv(plc, "title", "CDF"))
    plt.grid(plc.grid)
    if plc.legend:
        plt.legend()
    if plc.xlim is not None:
        plt.xlim(plc.xlim)
    if plc.ylim is not None:
        plt.ylim(plc.ylim)
    plt.show()


def plot_cdfs(plc, vals, cdfs):
    plt.figure(figsize=plc.figsize)
    labels = plcv(plc, "labels", [f"CDF {i}" for i in range(len(cdfs))])
    for i, cdf in enumerate(cdfs):
        (line,) = plt.plot(vals, cdf, linestyle="-", label=labels[i])
        color = line.get_color()
        plt.fill_between(vals, cdf, color=color, alpha=0.3)

    plt.xlabel(plcv(plc, "xlabel", "Epoch"))
    plt.ylabel(plcv(plc, "ylabel", "Loss"))
    plt.title(plcv(plc, "title", "CDF"))
    plt.grid(plc.grid)
    if plc.legend:
        plt.legend()
    if plc.xlim is not None:
        plt.xlim(plc.xlim)
    if plc.ylim is not None:
        plt.ylim(plc.ylim)
    plt.show()
