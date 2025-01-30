## Readme Template

A starter readme with basic info for interacting with a new rye repo.

Assumes you followed setup instructions from [dr_setup/repos](https://github.com/drothermel/dr_setup/blob/repo_setup/repos/README.md) which will provide:
- `.gitignore` with vim, python, jupyter support
- `rye` setup process
- `src/package/__init__.py` and `tests/...` dir to enable `rye test`

## Using Rye

Full deep dive: [dr_setup/repos/rye_usage.md](https://github.com/drothermel/dr_setup/blob/repo_setup/repos/rye_usage.md)

Ideally rye has already been setup:
```shell
rye init --no-readme --name PROJ_NAME REPO_PATH
rye pin 3.12
rye sync
rye show
rye list # jupyterlab, jupyterlab_vim, pytest, dr_util
```

Basic Usage:
```
rye add PACKAGE (opt: --dev) (opt: --git URL)
rye remove PACKAGE (opt: --dev)
rye fmt
rye lint --fix
rye test -v
rye run TOOL_NAME
```

## Using hydra + wandb

- Hydra deep dive: [dr_setup/repos/hydra_usage.md](https://github.com/drothermel/dr_setup/blob/repo_setup/repos/hydra_usage.md)
- Wandb deep dive: [dr_setup/repos/wandb_usage.md
](https://github.com/drothermel/dr_setup/blob/repo_setup/repos/wandb_usage.md)

Install both:
```shell
rye add wandb
rye add hydra-core==1.3
```

Use both in a script:
```python
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="qmp_cfg")
def run(cfg: DictConfig):
  logging.info(">> Input config yaml:")
  logging.info("\n" + str(OmegaConf.to_yaml(cfg)))
	
  cfg_dict = OmegaConf.to_container(cfg, resolve=True)
  wandb.init(
    project='name',
    config=cfg_dict,
  )
  wandb.log({"data": 100})

if __name__ == "__main__":
  run()
```
