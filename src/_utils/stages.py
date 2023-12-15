from pydantic import BaseModel
from typing import Dict, List, Union, Optional, Any
from _preprocessing.Preprocessor import Preprocessor
from _utils.enumerations import *


class PreprocessorConfig(BaseModel):
    preprocessor: PreprocessorEnum
    config: Dict


class NoTune(BaseModel):
    no_tune: bool


class EmptyDict(BaseModel):
    random_seed: int
    search_method_names: List[SearchMethodNoneEnum]
    no_tune: NoTune


class DatasetConfig(BaseModel, extra="allow"):
    name: str


class TuneConfig(BaseModel):
    search_method_name: Union[SearchMethodNoneEnum, TLTunerEnum]
    train_tune_option: TrainTuneOptionsEnum
    metric_key: Union[None, MetricKeysEnum]

    def set_search_method_name(
        self, search_method_name: Union[SearchMethodNoneEnum, TLTunerEnum]
    ) -> None:
        self.search_method_name = search_method_name
        self.check_attributes_coherence()

    def check_attributes_coherence(self):
        # check if None is used with train_configs_only
        assert (
            self.search_method_name == SearchMethodNoneEnum.none
            and self.train_tune_option == TrainTuneOptionsEnum.train_configs_only
        ) or (
            self.search_method_name != SearchMethodNoneEnum.none
            and self.train_tune_option != TrainTuneOptionsEnum.train_configs_only
        ), "When using train_configs_only option u need none as search_method_name"


class RayConfig(BaseModel):
    resources_per_trial: Dict[RayRessourcesEnum, int]
    name: str
    num_samples: int
    verbose: int
    log_to_file: bool
    raise_on_failed_trial: bool
    resume: Union[bool, str]
    time_budget: float
    local_dir: str = None
    time_budget_s: int = None
    fail_fast: Union[bool, str]
    max_concurrent_trials: int

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.time_budget_s = int(self.time_budget * 3600)


class PreprocessStage(BaseModel):
    dataset_name: DatasetNameEnum
    dataset_config: Dict[DomainNameEnum, DatasetConfig]
    preprocessing_pipeline: List[PreprocessorConfig]
    random_seed: Optional[int]


class NoShuffleSplitConfig(BaseModel):
    n_splits: int
    test_size: float


class TrainStage(BaseModel):
    classifier_name: TLClassifierEnum
    config: Dict
    tune_config: TuneConfig
    random_seed: Optional[int]
    no_shuffle_split_config: NoShuffleSplitConfig


class ModelSelection(BaseModel):
    name: ModelSelectionEnum
    config: Dict


class TunerConfig(BaseModel):
    hyperparam_tuning: Dict  # hparams that will be tuned via ray
    hyperparam_fixed: Dict  # hparams that will not be tuned
    model_selection: ModelSelection
    address: str = None


class TuneStage(BaseModel):
    classifier_name: TLClassifierEnum
    search_method_names: List[TLTunerEnum]
    tuner_config: TunerConfig
    ray_config: RayConfig
    random_seed: Optional[int]


class LoadModelConfig(BaseModel):
    checkpoint: Optional[Union[CheckpointLoadEnum, str]]
    metric_key: Optional[MetricKeysEnum]
    best: Optional[BestCheckpointEnum]
    on_domain: Optional[DomainNameEnum]


class PredictStage(BaseModel):
    load_model: LoadModelConfig
    random_seed: Optional[int]


class ScoreStage(BaseModel):
    random_seed: Optional[int]


class Stages(BaseModel):
    preprocess: PreprocessStage
    tune: Union[EmptyDict, TuneStage]
    train: TrainStage
    predict: PredictStage
    score: ScoreStage
