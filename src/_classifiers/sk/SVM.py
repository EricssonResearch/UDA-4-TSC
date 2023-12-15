from _classifiers.sk.base import SKClassifier
from _datasets.base import TLDataset
from sklearn.svm import SVC
from typing import Dict, TypeVar
from _utils.enumerations import DataSplitEnum


TSVM = TypeVar("TSVM", bound="SVM")


class SVM(SKClassifier):

    classifier: SVC

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, SVC, **kwargs)

    def fit(self, tl_dataset: TLDataset) -> TSVM:
        x_source, y_source = self.transform_dataset_to_X_Y_numpy(
            tl_dataset.source[DataSplitEnum.train]
        )

        self.classifier.fit(x_source, y_source)

        print("Fitted support vector classifier")

        return self
