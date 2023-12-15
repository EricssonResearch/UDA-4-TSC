from typing import TypeVar
import abc
from typing import Union
from _utils.enumerations import *
from _utils.stages import LoadModelConfig


TTLTuner = TypeVar("TTLTuner", bound="TLTuner")


class TLTuner(abc.ABC):
    def __init__(self, default_metric_key: MetricKeysEnum):
        self.default_metric_key = default_metric_key

    @abc.abstractmethod
    def get_metric_key(self, metric_key: MetricKeysEnum) -> str:
        """
        This should return the metric key to be used in get best configs
        """
        pass

    def get_metric_key_if_none(
        self, metric_name: Union[None, MetricKeysEnum]
    ) -> Union[None, MetricKeysEnum]:
        return metric_name if metric_name is not None else self.default_metric_key

    @abc.abstractmethod
    def fill_load_model(self, load_model_cfg: LoadModelConfig) -> None:
        """Will fill the corresonding configuration to know how to fetch best epoch"""
        # note that this is ignore and useless for none deep models
        raise NotImplementedError

    def unfill_load_model(self, load_model_cfg: LoadModelConfig) -> None:
        """Will undo the fill_load_model"""
        load_model_cfg.metric_key = None
        load_model_cfg.best = None
        load_model_cfg.on_domain = None
