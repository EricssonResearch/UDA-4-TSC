from _tuners.reweight.base import Reweight
from _utils.enumerations import *
from _utils.stages import LoadModelConfig


N_COMPONENTS = 5


class Reweight_5(Reweight):
    def __init__(self):
        super().__init__(
            n_components=N_COMPONENTS, default_metric_key=MetricKeysEnum.reweight_5_loss
        )

    def fill_load_model(self, load_model_cfg: LoadModelConfig) -> None:
        if load_model_cfg.metric_key is None:
            load_model_cfg.metric_key = MetricKeysEnum.reweight_5_loss
            load_model_cfg.best = BestCheckpointEnum.minimum
            load_model_cfg.on_domain = DomainNameEnum.source
