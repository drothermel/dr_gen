# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# %% Make Mock Cifar Data
def make_mock_cifar_df(
        lrs=(0.01, 0.03, 0.1),
        wds=(1e-4, 3e-4, 1e-3),
        n_epochs=100,
        seed=7,
) -> pd.DataFrame:
    """
    Create synthetic ResNet‑18 training curves that *look* like the ones in the
    screenshot.  The exact numerical values are arbitrary, but the trends are
    realistic and clearly separated by (lr, wd).

    Returns
    -------
    df : tidy DataFrame with columns
        epoch, lr, wd, train_loss, val_loss, train_acc, val_acc
    """
    rng = np.random.default_rng(seed)
    epochs = np.arange(1, n_epochs + 1)

    rows = []
    for lr in lrs:
        for wd in wds:
            # scaling factors: higher lr ⇒ steeper loss decay; higher wd ⇒ more penalty
            lr_factor = np.log10(lr / min(lrs))          # 0.0, 0.477…, 1.0 for the default list
            wd_factor = np.log10(wd / min(wds))          # 0.0, 0.477…, 1.0

            for e in epochs:
                log_e = np.log10(e)

                # mock losses
                base_train_loss = 2.6 - 0.35 * log_e                 # common decay
                train_loss = base_train_loss \
                             - 0.08 * lr_factor + 0.04 * wd_factor \
                             + rng.normal(0, 0.01)

                val_loss = train_loss + 0.12 + rng.normal(0, 0.015)

                # mock accuracies (start low, rise slowly)
                base_acc = 0.15 + 0.18 * log_e                       # common growth
                train_acc = base_acc \
                            + 0.03 * lr_factor - 0.02 * wd_factor \
                            + rng.normal(0, 0.005)

                val_acc = train_acc - 0.05 + rng.normal(0, 0.005)

                rows.append(
                    dict(epoch=e, lr=lr, wd=wd,
                         train_loss=train_loss, val_loss=val_loss,
                         train_acc=train_acc, val_acc=val_acc)
                )

    return pd.DataFrame(rows)

# %% Plotting Function
def plot_training_metrics(df: pd.DataFrame) -> None:
    """
    Draw the *four* panels (train‑loss, val‑loss, train‑acc, val‑acc) with
    colour = learning‑rate and linestyle = weight‑decay.  The mapping is derived
    automatically from whatever unique values appear in the dataframe.

    Parameters
    ----------
    df : tidy dataframe with at least the columns produced by
         `make_mock_cifar_df`.
    """
    # ----- discover the hyper‑parameter palette / style -----------------
    lrs = sorted(df['lr'].unique())         # e.g. [0.01, 0.03, 0.1]
    wds = sorted(df['wd'].unique())         # e.g. [1e‑4, 3e‑4, 1e‑3]

    # colours – reuse tab10 but subsample in case there are more than 10 lrs
    cmap = plt.get_cmap('tab10')
    colour_map = {lr: cmap(i % 10) for i, lr in enumerate(lrs)}

    # linestyles – cycle through the classic trio, then fallback to dash‑dot etc.
    style_cycle = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1))]
    style_map = {wd: style_cycle[i % len(style_cycle)] for i, wd in enumerate(wds)}

    # ----- set‑up the 1×4 sub‑plots ------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharex=True)

    panels = [('train_loss', 'Train Loss', True),
              ('val_loss',   'Val Loss',   True),
              ('train_acc',  'Train Acc',  False),
              ('val_acc',    'Val Acc',    False)]

    for (metric, title, ylog), ax in zip(panels, axes):
        for lr in lrs:
            for wd in wds:
                subset = df[(df.lr == lr) & (df.wd == wd)]
                ax.plot(subset['epoch'], subset[metric],
                        label=f'lr={lr}, wd={wd}',
                        color=colour_map[lr],
                        linestyle=style_map[wd],
                        linewidth=1.8)

        ax.set_title(title)
        ax.set_xlabel('Epoch (log scale)')
        ax.grid(alpha=.3, which='both', linestyle=':')
        ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
            ax.set_ylabel('Loss (log scale)')
        else:
            ax.set_ylabel('Accuracy')

    # -------------------  Legend (row-per-lr layout)  --------------------

    title = 'Learning Rate (color) × Weight Decay (style)'

    handles, labels = [], []

    for lr in lrs:
        # first column of the row: lr label (no visible line)
        handles.append(Line2D([], [], color='none', label=f'lr={lr}:'))
        labels.append(f'lr={lr}:')
        # remaining columns: one entry per wd, styled correctly
        for wd in wds:
            handles.append(Line2D([], [], color=colour_map[lr],
                                linestyle=style_map[wd], linewidth=2,
                                label=f'wd={wd}'))
            labels.append(f'wd={wd}')

    # ncol = (# wds + 1)   →  exactly one row per lr group
    ncol = len(wds) + 1

    leg = fig.legend(handles, labels,
                    ncol=ncol,
                    loc='lower center',
                    bbox_to_anchor=(0.5, -0.10),
                    frameon=False,
                    columnspacing=1.5,
                    handletextpad=0.6)

    # add a bold title and subtitle (two separate lines)
    leg.set_title(f'{title}', prop={'weight': 'bold', 'size': 'medium'})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)   # leave room beneath the plots
    plt.show()

# %%
mock_df = make_mock_cifar_df()
mock_df.head()

# %% Plot the mock data
plot_training_metrics(mock_df)



# %%


































