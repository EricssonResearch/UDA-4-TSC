from transformers import EvalPrediction
from typing import Dict, Callable
from _utils.enumerations import *
from sklearn.metrics import classification_report
from _datasets.base import TLDataset
from _scoring.reweight.base import ReweightMetric


DIGITS = 4


class Scorer(Callable):
    """
    This class is able to combine several metrics in the evaluate library of HuggingFace
    """

    def __init__(self, tl_dataset: TLDataset):
        self.digits = DIGITS
        self.target_names = tl_dataset.get_target_names()
        self.reweight = ReweightMetric(tl_dataset=tl_dataset)

    def fit_gmm_if_needed(self, tl_dataset: TLDataset) -> None:
        """Will fit the gmm if needed"""
        tl_dataset.fit_gmm(self.reweight.n_components)

    def __call__(self, eval_prediction: EvalPrediction) -> Dict:
        probas = eval_prediction.predictions
        references = eval_prediction.label_ids
        predictions = probas.argmax(axis=1)

        res = classification_report(
            y_true=references,
            y_pred=predictions,
            digits=self.digits,
            output_dict=True,
            labels=[i for i in range(len(self.target_names))],
            target_names=self.target_names,
            zero_division=0,
        )

        # compute accuracy if not in report
        if MetricKeysEnum.accuracy.value not in res:
            res.update({MetricKeysEnum.accuracy.value: res["micro avg"]["f1-score"]})

        res.update(self.reweight.__call__(predictions=eval_prediction))

        return res

    def compute_metrics(
        self,
        predictions: Dict[DomainNameEnum, Dict[DataSplitEnum, EvalPrediction]],
    ) -> Dict[DomainNameEnum, Dict[DataSplitEnum, dict]]:
        """
        Compute for each domain (source & target) and for each split the metrics
        and return a dict.
        """
        # init the res dict
        res = {}

        # loop through predictions and compute accordingly
        for domain_name in predictions:
            res[domain_name] = {}
            for split in predictions[domain_name]:
                res[domain_name][split] = self(predictions[domain_name][split])

        return res
