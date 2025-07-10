# %%
print('Important HPMS:', db.important_hpms)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
display(db.active_runs_df.drop('seed', 'tag', 'optim.name', 'model.architecture').unique().select(['run_id','batch_size', 'model.width_mult', 'optim.lr', 'optim.weight_decay']).sort(pl.all()).to_pandas())

#db.config.use_runs_filters['no label smoothing'] = lambda run: run.hpms._flat_dict['train_transforms.label_smoothing'] == 0.0
#db.update_filtered_runs()
#print(f"Number of active runs (without label smoothing): {len(db.active_runs)}")


# %%
def get_run_ids(db: ExperimentDB, batch_size: int, width_mult: float) -> list[str]:
    return db.active_runs_df.filter(
        (pl.col("batch_size") == batch_size) & (pl.col("model.width_mult") == width_mult)
    ).select('run_id').to_series().to_list()

test_rids = get_run_ids(db, batch_size=128, width_mult=1.0)
print(len(test_rids), test_rids[:3])


#db.active_metrics_df().head()
db._metrics_df.filter(pl.col('run_id').is_in(test_rids)).head()


# %%

metrs_t0 = db.query_metrics('train_loss', run_filter=test_rids)
metrs_t0.head()





# %%
db._metrics_df.head()


mets_t0 = group_metric_by_hpms_v2(
    db, 'train_loss', **{'batch_size':128, 'model.width_mult': 1.0},
)