from typing import Callable
from _utils.enumerations import *
from _datasets.base import TLDataset
from transformers import EvalPrediction
from typing import Dict
import numpy as np
from scipy.special import xlogy
from sklearn.preprocessing import OneHotEncoder
import sys
from ray.tune.utils.util import is_nan_or_inf
from pydantic import BaseModel


class ResultReweight(BaseModel):
    reweighted_loss: float = np.nan
    reweighted_target_density_loss: float = np.nan


class ReweightMetric(Callable[[str], float]):
    def __init__(
        self,
        tl_dataset: TLDataset,
    ):
        self.n_components = 5
        self.tl_dataset = tl_dataset
        self.label_binarizer: OneHotEncoder = self.tl_dataset.get_label_binarizer()
        self._density_at_source_point = None

    def _get_density_at_source_point(self) -> Dict[DomainNameEnum, np.ndarray]:
        """
        Compute for once the value of the density at each source point.
        This needs to be computed once and can be cached here
        assuming the dataset is constant and not changing once the scorer is created
        note that this will create issues when doing cross validation however we are not
        supporting this for now
        """
        if self._density_at_source_point is not None:
            return self._density_at_source_point
        self._density_at_source_point = self.tl_dataset.get_density_at_source_point(
            n_components=self.n_components
        )
        return self._density_at_source_point

    def __call__(self, predictions: EvalPrediction, **kwargs) -> Dict[str, float]:
        """Computes the source reweighted loss, usually it only make sense for source.test set"""
        res = self.reweight_loss(
            prediction_source=predictions,
        )
        return {
            MetricKeysEnum.reweight_5_loss.value: res.reweighted_loss,
            MetricKeysEnum.reweight_5_target_density_loss.value: res.reweighted_target_density_loss,
        }

    def compute_cross_entropy_per_example(self, eval_preds: EvalPrediction) -> np.array:
        # get labels into one hot
        label_ids = eval_preds.label_ids
        label_ids = self.label_binarizer.transform(label_ids[:, None])
        # get prediction and clip to avoid numeric instability
        y_pred = eval_preds.predictions
        eps = np.finfo(y_pred.dtype).eps
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # compute the cross entropy
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
        loss_per_example = -xlogy(label_ids.astype(float), y_pred).sum(axis=1)
        # return
        return loss_per_example

    def reweight_loss(
        self,
        prediction_source: EvalPrediction,
    ) -> ResultReweight:
        """
        Compute and reweight the loss source data
        using the ratio of the density between the target and source distribution.
        """
        if self._check_if_should_not_compute(prediction=prediction_source):
            return ResultReweight()

        # loss per example
        loss_per_example = self.compute_cross_entropy_per_example(prediction_source)

        # get the dict
        density_at_source_point = self._get_density_at_source_point()

        # fetch the source/target density at source point
        source_density_at_source_point = density_at_source_point[DomainNameEnum.source]
        target_density_at_source_point = density_at_source_point[DomainNameEnum.target]

        # compute the density ratio between source and target
        density_ratio = target_density_at_source_point / source_density_at_source_point

        # average of the reweight loss
        reweight_loss = (loss_per_example * density_ratio).mean()

        # fill if NaN or Inf
        reweight_loss = sys.float_info.max if is_nan_or_inf(reweight_loss) else reweight_loss

        # target loss estimation using only source point
        reweight_target_loss_at_source_point = (
            loss_per_example * target_density_at_source_point
        ).mean()

        # fill if NaN or Inf
        reweight_target_loss_at_source_point = (
            sys.float_info.max
            if is_nan_or_inf(reweight_target_loss_at_source_point)
            else reweight_target_loss_at_source_point
        )

        return ResultReweight(
            reweighted_loss=reweight_loss,
            reweighted_target_density_loss=reweight_target_loss_at_source_point,
        )

    def _check_if_should_not_compute(self, prediction: EvalPrediction) -> bool:
        """
        since this metric only makes sense for source test set avoid errors
        first check if the size mismatch
        then check if they have exactly the same labels in the same order
        this is mainly to avoid errors even if this metric is computed for other than source.test
        it should be ignored when viewing the results
        """

        if len(prediction.label_ids) != len(self.tl_dataset.source[DataSplitEnum.test]):
            return True

        if not (
            np.asarray(prediction.label_ids)
            == np.asarray(self.tl_dataset.source[DataSplitEnum.test][DatasetColumnsEnum.labels])
        ).all():
            return True

        return False
