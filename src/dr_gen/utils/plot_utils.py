from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np


def get_plt_cfg(**kwargs):
    base_plt_cfg = OmegaConf.create({
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
    })
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
    labels = plcv(plc, 'labels', range(len(data_lists)))
    x = range(len(data_lists[0]))
    for i, data in enumerate(data_lists):
        plt.plot(x, data, linestyle='-', label=labels[i])
        
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

def plot_summary_lines(
    plc, 
    data_lists,
):
    if len(data_lists) == 0 or len(data_lists[0]) == 0:
        print(">> invalid data")
        return
    plt.figure(figsize=plc.figsize)
    labels = plcv(plc, 'labels', ["Mean"])
    x = range(len(data_lists[0]))
    v_array = np.array([dl for dl in data_lists if len(dl) == len(x)])
    v_mean = np.mean(v_array, axis=0)
    v_std = np.std(v_array, axis=0)
    v_min = np.min(v_array, axis=0)
    v_max = np.max(v_array, axis=0)

    line_mean, = plt.plot(x, v_mean, linewidth=3, label=labels[0])
    color = line_mean.get_color()
    plt.fill_between(x, v_min, v_max, color=color, alpha=0.1)
    plt.fill_between(
        x, v_mean - v_std, v_mean + v_std,
        color=color, alpha=0.3
    )
    plt.plot(
        x, v_mean - v_std, linestyle='-', linewidth=1,
        color=color, alpha=0.5, 
    )
    plt.plot(
        x, v_mean + v_std, linestyle='-', linewidth=1,
        color=color, alpha=0.5, 
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
    
        

    