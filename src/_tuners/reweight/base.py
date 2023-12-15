from _tuners.base import TLTuner
from _datasets.base import TLDataset
from _classifiers.base import TLClassifier
from typing import Dict
from _utils.enumerations import *
from _classifiers.sk.base import SKClassifier
from datasets import Dataset

import numpy as np
from sklearn.mixture import GaussianMixture
from transformers import EvalPrediction
from scipy.special import xlogy
import sys
from ray.tune.utils.util import is_nan_or_inf


class Reweight(TLTuner):
    """
    @article{sugiyama2007covariate,
    title={Covariate shift adaptation by importance weighted cross validation.},
    author={Sugiyama, Masashi and Krauledat, Matthias and M{\"u}ller, Klaus-Robert},
    journal={Journal of Machine Learning Research},
    volume={8},
    number={5},
    year={2007}
    }
    """

    def __init__(self, n_components: int, default_metric_key: MetricKeysEnum):
        super().__init__(default_metric_key=default_metric_key)
        self.n_components = n_components

    def get_metric_key(self, metric_name: MetricKeysEnum) -> str:
        cur_metric_name = self.get_metric_key_if_none(metric_name=metric_name)
        return (
            f"{DomainNameEnum.source}/{AverageEnum.average}/{DataSplitEnum.test}/{cur_metric_name}"
        )
