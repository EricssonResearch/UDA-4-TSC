from _classifiers.sk.base import SKClassifier
from _datasets.base import TLDataset
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, TypeVar
from _utils.enumerations import DataSplitEnum


TRandomForest = TypeVar("TRandomForest", bound="RandomForest")


class RandomForest(SKClassifier):

    classifier: RandomForestClassifier

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, RandomForestClassifier, **kwargs)

    def fit(self, tl_dataset: TLDataset) -> TRandomForest:
        x_source, y_source = self.transform_dataset_to_X_Y_numpy(
            tl_dataset.source[DataSplitEnum.train]
        )

        self.classifier.fit(x_source, y_source)

        print("Fitted random forest")

        return self
