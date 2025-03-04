import hydra
from omegaconf import DictConfig

from dr_gen.analyze.log_file_data import LogFileData


@hydra.main(version_base=None, config_path="../configs", config_name="analyze")
def run(cfg: DictConfig):
    print()
    print(f">> Analyze: {cfg.log_file}")
    print()

    if cfg.log_file is None:
        print(">> No log file means nothing to analyze, set cfg.log_file")
        return

    lfd = LogFileData(cfg.log_file)
    print(f">> Parse Errors: {len(lfd.parse_errors)}")
    for pe in lfd.parse_errors:
        print(f"  - {pe}")
    print()

    print(">> Flat Config:")
    for k, v in lfd.get_flat_config().items():
        print(f" - {k:50} | {v}")
    print()

    print(">> All xs:")
    for k, v in lfd.get_all_xs_flat().items():
        print(f"  - {str(k):50} | {v[:3]}")
    print()

    print(">> All vals:")
    for k, v in lfd.get_all_vals_flat().items():
        print(f"  - {str(k):50} | {v[:3]}")
    print()

    print(">> End run!")


if __name__ == "__main__":
    run()
