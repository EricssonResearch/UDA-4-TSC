import hydra
import os
import pathlib
from glob import glob
from typing import Set
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from _utils.enumerations import *
from _utils.stages import Stages
from tqdm import tqdm


class TensorboardConfig(BaseModel):

    classifiers: Set[TLClassifierEnum]
    datasets: Set[DatasetNameEnum]
    tuners: Set[TLTunerEnum]
    out_dir: str
    xps_dir: str


def check_in_set(item: str, the_set: Set[str]):
    """Will return true if set is empty"""
    if len(the_set) == 0:
        return True
    else:
        return item in the_set


@hydra.main(version_base=None, config_path="../_configs/tensorboard", config_name="config")
def main(cfg: DictConfig) -> None:
    # parse config
    cfg: TensorboardConfig = TensorboardConfig(**OmegaConf.to_object(cfg))

    # create the dir if not exists
    os.makedirs(cfg.out_dir, exist_ok=True)

    # delete the files in this dir
    [pathlib.Path(f).unlink() for f in glob(f"{cfg.out_dir}/*")]

    # get globs
    pths = glob(f"{cfg.xps_dir}/*/")

    # loop through the xps
    for xp_pth in tqdm(pths, total=len(pths), desc=f"Creating symlinks in {cfg.out_dir}"):
        # get pipeline config
        params_file = f"{xp_pth}src/params.json"
        # skip if not exist (mostly xp currently running)
        if not os.path.exists(params_file):
            continue
        # parse config from params file
        stages: Stages = Stages.parse_file(params_file)

        # get useful vars
        dataset_name = stages.preprocess.dataset_name
        classifier_name = stages.train.classifier_name
        source_name = stages.preprocess.dataset_config[DomainNameEnum.source].name
        target_name = stages.preprocess.dataset_config[DomainNameEnum.target].name

        # skip if not in datasets or classifiers
        if not check_in_set(dataset_name, cfg.datasets) or not check_in_set(
            classifier_name, cfg.classifiers
        ):
            continue
        # loop through possible tuners
        for tuner in cfg.tuners:
            # get path of train tensorboard logs
            logs_pth = glob(f"{xp_pth}src/output/train/*/*/*/*/*{tuner}/runs/*/")

            if len(logs_pth) == 0:
                # skip most likely a clf not huggingface
                continue
            elif len(logs_pth) == 1:
                # good found exactly a single file
                pass
            else:
                # no good this should not happen
                raise Exception(f"several were found: {logs_pth}")

            # symlink name
            sym_lnk = f"{cfg.out_dir}/{dataset_name}[{source_name}-{target_name}]-{classifier_name}-{tuner}"

            # create symbolic link
            os.symlink(logs_pth[0], f"{sym_lnk}")


if __name__ == "__main__":
    main()
