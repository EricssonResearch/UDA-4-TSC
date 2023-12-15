from _utils.stages import Stages, DatasetConfig
from _utils.paths import Paths
import shutil, os, json, yaml
from pydantic import BaseModel
from _classifiers.base import TLClassifier
from typing import List, Dict, Tuple
from _utils.paths import Paths
from _utils.enumerations import DomainNameEnum, EnvVariables, DatasetNameEnum, TLClassifierEnum
import hydra
from omegaconf import DictConfig, OmegaConf


class UtilsConfig(BaseModel):
    force: bool
    localParams: str
    idxExperiment: int


class HydraConfigClass(BaseModel):
    utils: UtilsConfig
    stages: Stages


class SampleConfig(BaseModel):
    dataset_configs: List[Dict[DomainNameEnum, DatasetConfig]]


class DVCStage(BaseModel):

    """
    Should encapsulate this command https://dvc.org/doc/command-reference/stage/add
    """

    name: str
    params: List[str]
    outs: List[str]
    deps: List[str]
    force: bool
    cmd: str
    metrics_no_cache: List[str] = []
    create_gitignore: bool = True

    def add_to_yaml(self, stage_out_dir: str, gitignore_file: str):
        """
        Adds the stage to the dvc.yaml file
        """

        if self.create_gitignore:
            # create the output_dir folder if not exists
            # needed with gitignore in order for dvc to run

            # check if already exists
            if not os.path.exists(gitignore_file):
                # create the dir to make sure it exists
                os.makedirs(stage_out_dir, exist_ok=True)
                # write in it * to ignore everything
                with open(gitignore_file, "w") as f:
                    f.write("*/")

        # add the dvc stage
        full_cmd: str = (
            f"dvc stage add --name {self.name} " "--params " + ",".join(self.params) + " "
        )

        # add deps
        for out in self.deps:
            full_cmd = full_cmd + " --deps " + out

        # add outs
        for out in self.outs:
            full_cmd = full_cmd + " --outs " + out

        # add metrics if exist
        for metric_no_path in self.metrics_no_cache:
            full_cmd = full_cmd + " --metrics-no-cache " + metric_no_path

        # add force if needed
        if self.force:
            full_cmd = f"{full_cmd} --force "

        # add run command
        full_cmd = f"{full_cmd} {self.cmd}"

        # execute the command
        code = os.system(full_cmd)

        # exit if needed and raise the error code
        if code != 0:
            exit(code)


def parse_hydra(obj_cfg: HydraConfigClass) -> Stages:
    # parse the default params config file
    stages_config: Stages = obj_cfg.stages

    return stages_config


def dump_dvc_params_file(cfg: DictConfig) -> HydraConfigClass:
    """
    Dump the config file for the pipeline
    """
    obj_cfg: HydraConfigClass = HydraConfigClass(**OmegaConf.to_object(cfg))

    stages_config = parse_hydra(obj_cfg)

    # get paths object
    paths = Paths(stages_config=stages_config)

    # get the path to the sample config
    path_to_sample_config_file = paths.get_dataset_config_sample_file()

    # parse the sample config file
    sample_config: SampleConfig = SampleConfig.parse_file(path_to_sample_config_file)

    # update the stages_config from sample_config
    # get the dataset config that we will use to update the stages config
    update_dataset_config = sample_config.dataset_configs[obj_cfg.utils.idxExperiment]
    # update the source config dataset
    source_config = stages_config.preprocess.dataset_config[DomainNameEnum.source].dict()
    source_config.update(update_dataset_config[DomainNameEnum.source])
    stages_config.preprocess.dataset_config[DomainNameEnum.source] = DatasetConfig(**source_config)
    # update the target config dataset
    target_config = stages_config.preprocess.dataset_config[DomainNameEnum.target].dict()
    target_config.update(update_dataset_config[DomainNameEnum.target])
    stages_config.preprocess.dataset_config[DomainNameEnum.target] = DatasetConfig(**target_config)

    # dump the new stages_config in local
    with open(obj_cfg.utils.localParams, "w") as f:
        json.dump(stages_config.dict(), f, indent=4)

    print(f"Generated {obj_cfg.utils.localParams}")

    return obj_cfg


@hydra.main(version_base=None, config_path="../_configs/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Create the dvc stages in dvc.yaml file and the params file
    """

    obj_cfg = dump_dvc_params_file(cfg)


if __name__ == "__main__":
    main()
