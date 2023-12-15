from _tuners.base import TLTuner
from _utils.enumerations import *
from _utils.stages import LoadModelConfig


class NoTune(TLTuner):
    def __init__(self):
        super().__init__(default_metric_key=MetricKeysEnum.accuracy)

    def get_metric_key(self, **kwargs) -> None:
        raise Exception("This should never be called")

    def fill_load_model(self, load_model_cfg: LoadModelConfig) -> None:
        if load_model_cfg.metric_key is None:
            load_model_cfg.metric_key = MetricKeysEnum.accuracy
            load_model_cfg.best = BestCheckpointEnum.maximum
            load_model_cfg.on_domain = DomainNameEnum.source
