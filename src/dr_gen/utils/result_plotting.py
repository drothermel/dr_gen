import dr_gen.utils.result_parsing as rp
import dr_gen.utils.plot_utils as pu

def get_run_sweep_kvs(run_logs, combo_key_order, ind):
    run_cfg, _, _ = run_logs[ind]
    kvs = []
    for k in combo_key_order:
        kvs.append((k, run_cfg[k]))
    kv_strs = []
    k_strs = [k.split('.')[-1] for k in combo_key_order]
    v_strs = []
    for k, v in kvs:
        kstr = k.split('.')[-1]
        if isinstance(v, int):
            vstr = f"{int(v)}"
        elif isinstance(v, float):
            vstr = f"{float(v):0.1e}"
        else:
            vstr = str(v)
        kv_strs.append(f"{kstr}={vstr}")
    return kvs, " ".join(kv_strs)
    

def plot_run_splits(
    runs,
    remapped_metrics,
    sweep_info,
    run_ind,
    splits=['train', 'val', 'eval'],
    metric="acc1",
    **kwargs,
):
    _, kvstr = get_run_sweep_kvs(
        runs, sweep_info['combo_key_order'], run_ind,
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
                remapped_metrics, split, metric, run_ind,
            ) for split in splits
        ],
    )
    
    