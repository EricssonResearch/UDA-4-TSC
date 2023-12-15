from _tuners.base import TLTuner
from typing import TypeVar
from _utils.enumerations import *
from _utils.stages import LoadModelConfig


class TransportedTarget(TLTuner):
    def __init__(self):
        super().__init__(default_metric_key=MetricKeysEnum.accuracy)

    def get_metric_key(self, metric_name: MetricKeysEnum) -> str:
        cur_metric_name = self.get_metric_key_if_none(metric_name=metric_name)
        return (
            f"{DomainNameEnum.target}/{AverageEnum.average}/{DataSplitEnum.test}/{cur_metric_name}"
        )

    def fill_load_model(self, load_model_cfg: LoadModelConfig) -> None:
        if load_model_cfg.metric_key is None:
            load_model_cfg.metric_key = MetricKeysEnum.accuracy
            load_model_cfg.best = BestCheckpointEnum.maximum
            load_model_cfg.on_domain = DomainNameEnum.target
