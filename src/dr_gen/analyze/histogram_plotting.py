import matplotlib.pyplot as plt
from dr_gen.analyze.plot_utils import plcv, get_plt_cfg

def plot_histogram(
    plc,
    values,
):
    plt.figure(figsize=plc.figsize)
    plt.hist(
        values,
        bins=plc.nbins, edgecolor='black', range=plc.hist_range,
        density=plc.density,
    )
    plt.xlabel(plcv(plc, "xlabel", "acc1"))
    plt.ylabel(plcv(plc, "ylabel", "Frequency"))
    plt.title(plcv(plc, "title", "Histogram"))
    plt.grid(plc.grid)
    plt.show()

def plot_histogram_compare(
    plc,
    ind_stats_list,
):
    plt.figure(figsize=plc.figsize)
    
    # Group 1
    colors = ['blue', 'red', 'black', 'purple']
    for i, ind_stats in enumerate(ind_stats_list):
        plt.hist(
            ind_stats['vals'],
            bins=plc.nbins, histtype='step', edgecolor=colors[i], 
            range=plc.hist_range,
            density=True,
        )
        plt.axvline(
            ind_stats['mean'],
            color=colors[i], linestyle='dashed', linewidth=1.5,
            label=f'{plc.labels[i]} Mean: ({ind_stats["mean"]:.2f})'
        )

    plt.xlabel(plcv(plc, "xlabel", "acc1"))
    plt.ylabel(plcv(plc, "ylabel", "Frequency"))
    plt.title(plcv(plc, "title", f"Histogram"))
    if plc.legend:
        plt.legend()
    plt.grid(plc.grid)
    plt.show()
    
