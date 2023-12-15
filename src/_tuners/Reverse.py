from _tuners.base import TLTuner
from _datasets.base import TLDataset
from _classifiers.base import TLClassifier
from _utils.enumerations import *
from typing import Dict
import numpy as np
from datasets import Dataset, DatasetDict


class Reverse(TLTuner):
    def __init__(self):
        super().__init__(default_metric_key=MetricKeysEnum.accuracy)

    def evaluation(
        self,
        tl_classifier: TLClassifier,
        tl_dataset: TLDataset,
        classifier_hparams: Dict,
    ) -> Dict[TLTunerEnum, Dict]:
        """
        Zhong, Erheng, et al. 2010
        "Cross validation framework to choose amongst models and datasets for transfer learning."
        Only the reverse validation part that correspond to the section 3.1.
        """

        # predict the target train and test dataset label (called pseudo label)
        prediction = tl_classifier.predict(tl_dataset=tl_dataset)
        prediction_target_train = list(
            np.argmax(
                prediction[DomainNameEnum.target.value][DataSplitEnum.train.value].predictions,
                axis=1,
            )
        )
        prediction_target_test = list(
            np.argmax(
                prediction[DomainNameEnum.target.value][DataSplitEnum.test.value].predictions,
                axis=1,
            )
        )

        # define the two map functions that will update the real label with the pseudo label
        def update_target_label_with_pseudolabel(data, idx, prediction_target):
            data[DatasetColumnsEnum.labels.value] = prediction_target[idx].item()
            return data

        # define the new source train and test dataset using pseudo label
        new_source_train: Dataset = tl_dataset.target[DataSplitEnum.train.value].map(
            lambda data, idx: update_target_label_with_pseudolabel(
                data=data, idx=idx, prediction_target=prediction_target_train
            ),
            with_indices=True,
        )
        new_source_test: Dataset = tl_dataset.target[DataSplitEnum.test.value].map(
            lambda data, idx: update_target_label_with_pseudolabel(
                data=data, idx=idx, prediction_target=prediction_target_test
            ),
            with_indices=True,
        )

        # prepare the source dataset using the new source created
        # The test part should never used but we still update it.
        new_source: DatasetDict = DatasetDict(
            {
                DataSplitEnum.train.value: new_source_train,
                DataSplitEnum.test.value: new_source_test,
            }
        )

        # create the new tl dataset with the new source and the new target
        reverse_tl_dataset: TLDataset = TLDataset(
            dataset_name=tl_dataset.dataset_name,
            preprocessing_pipeline=tl_dataset.preprocessing_pipeline,
            source=new_source,
            target=tl_dataset.source,
        )

        # retrieve the class of the classifier
        classifier_class = type(tl_classifier)

        # define another classifier with the hyperpamater that we are looking at
        reverse_tl_classifier = classifier_class(
            classifier_hparams,
            train_dir="./tmp_trainer",  # no need to save load model when hparams tuning
            pred_dir="./tmp_trainer",  # no need to save load model when hparams tuning
        )

        # fit the reverse classifier with the new dataset
        reverse_tl_classifier = reverse_tl_classifier.fit(reverse_tl_dataset)

        # predict in the target label and compare it to the real label
        res = reverse_tl_classifier.evaluate(reverse_tl_dataset)

        return {TLTunerEnum.Reverse: res}

    def get_metric_key(self, metric_name: MetricKeysEnum) -> str:
        cur_metric_name = self.get_metric_key_if_none(metric_name=metric_name)
        return f"{TLTunerEnum.Reverse}/{AverageEnum.average}/{DomainNameEnum.target}/{DataSplitEnum.test}/{cur_metric_name}"
