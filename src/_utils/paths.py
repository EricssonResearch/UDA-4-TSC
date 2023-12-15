from pydantic import BaseModel
from _utils.enumerations import *
from _utils.stages import Stages
import os
import pathlib


# Maybe this should be a singleton class
class Paths(BaseModel):

    stages_config: Stages
    root_dir: str = "./output"
    root_config: str = "./_configs/custom_conf"
    data_root_dir: str = f"{root_dir}/data"
    train_root_dir: str = f"{root_dir}/train"
    tune_root_dir: str = f"{root_dir}/tune"
    pred_root_dir: str = f"{root_dir}/pred"
    results_root_dir: str = f"{root_dir}/results"
    aggregated_results_file: str = f"./_utils/results.jsonl"

    @staticmethod
    def get_env_variable(env_var_name: EnvVariables) -> str:
        if env_var_name in os.environ:
            return os.environ.get(env_var_name)
        else:
            raise Exception(f"Need to define {env_var_name}")

    @staticmethod
    def get_gmm_root_dir() -> str:
        gmm_root_dir = "./tmp_gmm"
        os.makedirs(gmm_root_dir, exist_ok=True)
        return str(pathlib.Path(gmm_root_dir).resolve())

    @staticmethod
    def get_fname_gmm(gmm_root_dir: str, domain_name: DomainNameEnum, n_components: int) -> str:
        return str(
            pathlib.Path(f"{gmm_root_dir}/tmp_gmm_{n_components}_{domain_name}.pkl").resolve()
        )

    def get_dataset_config_sample_file(self) -> str:
        return (
            f"{self.root_config}"
            f"/dataset_name={self.stages_config.preprocess.dataset_name}"
            f"/sample.json"
        )

    def get_dataset_disk_dir(self) -> str:
        return str(
            pathlib.Path(
                f"{self.data_root_dir}"
                f"/dataset_name={self.stages_config.preprocess.dataset_name.value.strip()}"
                f"/source={self.stages_config.preprocess.dataset_config[DomainNameEnum.source].name}"
                f"/target={self.stages_config.preprocess.dataset_config[DomainNameEnum.target].name}"
            ).resolve()
        )

    def get_dataset_disk_dir_source(self) -> str:
        return f"{self.get_dataset_disk_dir()}" f"/domain_name={DomainNameEnum.source.value}"

    def get_dataset_disk_dir_target(self) -> str:
        return f"{self.get_dataset_disk_dir()}" f"/domain_name={DomainNameEnum.target.value}"

    def get_tune_dir(self) -> str:
        return (
            f"{self.tune_root_dir}"
            f"/dataset_name={self.stages_config.preprocess.dataset_name.value}"
            f"/source={self.stages_config.preprocess.dataset_config[DomainNameEnum.source].name}"
            f"/target={self.stages_config.preprocess.dataset_config[DomainNameEnum.target].name}"
            f"/classifier_name={self.stages_config.train.classifier_name.value}"
        )

    def get_path_state_final(self, state_name: str) -> str:
        return self.get_tune_final() + "/" + state_name

    def get_tune_xp_path(self, tune_dir: str = None) -> str:
        tune_dir = self.get_tune_dir() if tune_dir is None else tune_dir
        return tune_dir + "/" + self.stages_config.tune.ray_config.name

    def get_tune_final(self, tune_dir: str = None) -> str:
        tune_dir = self.get_tune_dir() if tune_dir is None else tune_dir
        return tune_dir + "/final"

    def get_ray_df_pth(self, tune_dir: str = None) -> str:
        return self.get_tune_final(tune_dir=tune_dir) + "/ray_df.csv"

    def get_ray_best_cfgs_pth(self, tune_dir: str = None) -> str:
        return self.get_tune_final(tune_dir=tune_dir) + "/ray_best_cfg.json"

    def get_tune_eval_paths(self, tune_dir=None) -> str:
        return self.get_tune_xp_path(tune_dir=tune_dir) + "/evaluate_*/"

    def get_tune_state_xp(self, tune_dir=None) -> str:
        return self.get_tune_xp_path(tune_dir=tune_dir) + "/experiment_state*.json"

    def get_empty_tune_file(self) -> str:
        return f"{self.get_tune_final()}/empty.txt"

    def get_train_dir_dvc(self) -> str:
        return (
            f"{self.train_root_dir}"
            f"/dataset_name={self.stages_config.preprocess.dataset_name.value}"
            f"/source={self.stages_config.preprocess.dataset_config[DomainNameEnum.source].name}"
            f"/target={self.stages_config.preprocess.dataset_config[DomainNameEnum.target].name}"
            f"/classifier_name={self.stages_config.train.classifier_name.value}"
        )

    def get_train_dir(self) -> str:
        return (
            f"{self.get_train_dir_dvc()}"
            f"/search_method_name={self.stages_config.train.tune_config.search_method_name}"
        )

    def get_gitignore(self, root_folder: str) -> str:
        return f"{root_folder}/.gitignore"

    def get_pred_dir_dvc(self) -> str:
        return (
            f"{self.pred_root_dir}"
            f"/dataset_name={self.stages_config.preprocess.dataset_name.value}"
            f"/source={self.stages_config.preprocess.dataset_config[DomainNameEnum.source].name}"
            f"/target={self.stages_config.preprocess.dataset_config[DomainNameEnum.target].name}"
            f"/classifier_name={self.stages_config.train.classifier_name.value}"
        )

    def get_pred_dir(self) -> str:
        return (
            f"{self.get_pred_dir_dvc()}"
            f"/search_method_name={self.stages_config.train.tune_config.search_method_name}"
        )

    def get_pred_pkl(self) -> str:
        return f"{self.get_pred_dir()}/preds.pkl"

    def get_results_dir(self) -> str:
        result_dir = (
            f"{self.results_root_dir}"
            f"/dataset_name={self.stages_config.preprocess.dataset_name.value.strip()}"
            f"/source={self.stages_config.preprocess.dataset_config[DomainNameEnum.source].name}"
            f"/target={self.stages_config.preprocess.dataset_config[DomainNameEnum.target].name}"
            f"/classifier_name={self.stages_config.train.classifier_name.value}"
        )
        return result_dir

    def get_results_file(self) -> str:
        return f"{self.get_results_dir()}/metrics.json"

    def get_pipeline_file(self, stage_name: str) -> str:
        return f"_pipeline/{stage_name}.py"

    def get_dataset_path(self) -> str:
        return f"_datasets/{self.stages_config.preprocess.dataset_name}"
