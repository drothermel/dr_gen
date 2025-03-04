import dr_gen.analyze.result_parsing as rp
import dr_gen.analyze.plot_utils as pu
import dr_gen.analyze.histogram_plotting as hp
import dr_gen.analyze.ks_stats as ks


def kvs_to_str(kvs):
    kv_strs = []
    for k, v in kvs:
        kstr = k.split(".")[-1]
        if k == "model.weights":
            kstr = "init"
            vstr = "random" if v == "None" else "pretrain"
        else:
            vstr = str(v)
        kv_strs.append(f"{kstr}={vstr}")
    return " ".join(kv_strs)


def get_run_sweep_kvs(
    run_logs,
    combo_key_order,
    ind,
    seed=False,
    ignore_keys=[],
):
    keys = [*combo_key_order, "seed" if seed else ""]
    keys = [k for k in keys if k != "" and k not in ignore_keys]

    run_cfg, _, _ = run_logs[ind]
    kvs = [(k, run_cfg[k]) for k in keys]
    kv_str = kvs_to_str(kvs)
    return kvs, kv_str


def plot_run_splits(
    runs,
    remapped_metrics,
    sweep_info,
    run_ind,
    splits=["train", "val", "eval"],
    metric="acc1",
    ignore_keys=[],
    **kwargs,
):
    _, kvstr = get_run_sweep_kvs(
        runs,
        sweep_info["combo_key_order"],
        run_ind,
        seed=True,
        ignore_keys=ignore_keys,
    )
    plc_args = {
        "ylim": (70, 100),
        "labels": splits,
        "title": f"{metric} | {kvstr}",
        "ylabel": metric,
    }
    plc_args.update(kwargs)
    plc = pu.get_plt_cfg(
        **plc_args,
    )
    pu.plot_lines(
        plc,
        [
            rp.get_run_metrics(
                remapped_metrics,
                split,
                metric,
                run_ind,
            )
            for split in splits
        ],
    )


def plot_split_summaries(
    runs,
    remapped_metrics,
    sweep_info,
    kv_select,
    splits=["train", "val", "eval"],
    metric="acc1",
    ignore_keys=[],
    num_seeds=None,
    **kwargs,
):
    all_kvs, split_vals, _ = rp.get_selected_combo(
        runs,
        remapped_metrics,
        sweep_info,
        kv_select,
        splits,
        metric,
        ignore_keys,
        num_seeds,
    )
    assert len(all_kvs) == 1, ">> Only supports one combo for now"

    # Get the title from the kvs and num runs
    kv_str = kvs_to_str(all_kvs[0])
    num_seeds = len(split_vals[splits[0]][0])
    kv_str = f"{kv_str} | #Seeds: {num_seeds:,}"

    # Prepare plot
    plc_args = {
        "ylim": (70, 100),
        "labels": [f"{spl} mean {metric}" for spl in splits],
        "title": kv_str,
        "ylabel": metric,
    }
    plc_args.update(kwargs)
    pu.plot_summary_lines(
        pu.get_plt_cfg(**plc_args),
        [split_vals[split][0] for split in splits],
    )
    return


def plot_combo_histogram(
    runs,
    remapped_metrics,
    sweep_info,
    kv_select,
    split,
    epoch,
    metric="acc1",
    ignore_keys=[],
    num_seeds=None,
    **kwargs,
):
    all_kvs, all_split_vals, _ = rp.get_selected_combo(
        runs,
        remapped_metrics,
        sweep_info,
        kv_select,
        splits=[split],
        metric=metric,
        ignore_keys=ignore_keys,
        num_seeds=num_seeds,
    )

    all_ind_stats = []
    for split_vals in all_split_vals[split]:
        all_ind_stats.append(
            pu.get_runs_data_stats_ind(
                split_vals,
                ind=epoch,
            )
        )

    if len(all_ind_stats) > 1:
        print(f">> Just using first of {len(all_ind_stats)} combos")

    n = all_ind_stats[0]["n"]
    plc_args = {
        "nbins": n // 4,
        "hist_range": (80, 90),
        "title": f"Accuracy Distribution, {n} Seeds",
        "ylabel": "Num Runs",
    }
    plc_args.update(kwargs)
    plc = pu.get_plt_cfg(
        **plc_args,
    )

    hp.plot_histogram(
        plc,
        all_ind_stats[0]["vals"],
    )


def plot_combo_histogram_compare(
    runs,
    remapped_metrics,
    sweep_info,
    kv_select,
    split,
    epoch,
    metric="acc1",
    ignore_keys=[],
    num_seeds=None,
    vary_key="model.weights",
    **kwargs,
):
    all_kvs, all_split_vals, _ = rp.get_selected_combo(
        runs,
        remapped_metrics,
        sweep_info,
        kv_select,
        splits=[split],
        metric=metric,
        ignore_keys=ignore_keys,
        num_seeds=num_seeds,
    )

    all_ind_stats = []
    for split_vals in all_split_vals[split]:
        all_ind_stats.append(
            pu.get_runs_data_stats_ind(
                split_vals,
                ind=epoch,
            )
        )

    labels_kvstrs = [
        kvs_to_str([(k, v) for k, v in kvs if k == vary_key]) for kvs in all_kvs
    ]
    title_kvstr = kvs_to_str([(k, v) for k, v in all_kvs[0] if k != vary_key])
    ns = [sts["n"] for sts in all_ind_stats]
    plc_args = {
        "nbins": max(ns) // 4,
        "hist_range": (80, 90),
        "title": f"Accuracy Distribution | {title_kvstr}",
        "ylabel": "Num Runs",
        "labels": labels_kvstrs,
    }
    plc_args.update(kwargs)
    plc = pu.get_plt_cfg(
        **plc_args,
    )

    hp.plot_histogram_compare(plc, all_ind_stats)


def ks_stats_plot_cdfs(
    runs,
    remapped_metrics,
    sweep_info,
    kv_select,
    split,
    epoch,
    metric="acc1",
    ignore_keys=[],
    num_seeds=None,
    vary_key="model.weights",
    vary_vals=None,
    **kwargs,
):
    all_kvs, all_split_vals, _ = rp.get_selected_combo(
        runs,
        remapped_metrics,
        sweep_info,
        kv_select,
        splits=[split],
        metric=metric,
        ignore_keys=ignore_keys,
        num_seeds=num_seeds,
    )

    selected_kvs = []
    all_ind_vals = []
    for i, split_vals in enumerate(all_split_vals[split]):
        kv = all_kvs[i]
        v = [v for k, v in kv if k == vary_key]
        assert len(v) == 1
        v = v[0]
        if vary_vals is None or v in vary_vals:
            all_ind_vals.append(
                pu.get_runs_data_stats_ind(
                    split_vals,
                    ind=epoch,
                )["vals"]
            )
            selected_kvs.append(kv)

    assert len(all_ind_vals) == 2
    results = ks.calculate_ks_for_run_sets(
        all_ind_vals[0],
        all_ind_vals[1],
    )

    labels_kvstrs = [
        kvs_to_str([(k, v) for k, v in kvs if k == vary_key]) for kvs in selected_kvs
    ]
    title_kvstr = kvs_to_str([(k, v) for k, v in selected_kvs[0] if k != vary_key])
    ns = [len(vs) for vs in all_ind_vals]

    plc_args = {
        "labels": [
            f"{label} | #seed: {ns[i]}" for i, label in enumerate(labels_kvstrs)
        ],
        "title": f"CDF | {title_kvstr}",
    }
    plc_args.update(kwargs)
    plc = pu.get_plt_cfg(
        **plc_args,
    )
    pu.plot_cdfs(
        plc,
        results["all_vals"],
        [results["cdf1"], results["cdf2"]],
    )


def ks_stat_plot_cdfs_histograms(
    runs,
    remapped_metrics,
    sweep_info,
    kv_select,
    split,
    epoch,
    metric="acc1",
    ignore_keys=[],
    num_seeds=None,
    vary_key="model.weights",
    vary_vals=None,
    **kwargs,
):
    all_kvs, all_split_vals, _ = rp.get_selected_combo(
        runs,
        remapped_metrics,
        sweep_info,
        kv_select,
        splits=[split],
        metric=metric,
        ignore_keys=ignore_keys,
        num_seeds=num_seeds,
    )

    selected_kvs = []
    all_ind_stats = []
    for i, split_vals in enumerate(all_split_vals[split]):
        kv = all_kvs[i]
        v = [v for k, v in kv if k == vary_key]
        assert len(v) == 1
        v = v[0]
        if vary_vals is None or v in vary_vals:
            all_ind_stats.append(
                pu.get_runs_data_stats_ind(
                    split_vals,
                    ind=epoch,
                )
            )
            selected_kvs.append(kv)

    assert len(all_ind_stats) == 2
    results = ks.calculate_ks_for_run_sets(
        all_ind_stats[0]["vals"],
        all_ind_stats[1]["vals"],
    )

    labels_kvstrs = [
        kvs_to_str([(k, v) for k, v in kvs if k == vary_key]) for kvs in selected_kvs
    ]
    title_kvstr = kvs_to_str([(k, v) for k, v in selected_kvs[0] if k != vary_key])
    ns = [len(vs) for vs in all_ind_stats]

    # Plot the CDFs
    plc_args = {
        "labels": labels_kvstrs,
    }
    plc_args.update(kwargs)
    plc = pu.get_plt_cfg(
        **plc_args,
    )
    pu.plot_cdfs(
        plc,
        results["all_vals"],
        [results["cdf1"], results["cdf2"]],
    )

    # Plot the histograms
    plc_args = {
        "nbins": max(ns) // 4,
        "hist_range": (80, 90),
        "title": f"Accuracy Distribution | {title_kvstr}",
        "ylabel": "Num Runs",
        "labels": labels_kvstrs,
        "density": True,
    }
    plc_args.update(kwargs)
    plc = pu.get_plt_cfg(
        **plc_args,
    )

    hp.plot_histogram_compare(plc, all_ind_stats)
