from _classifiers.sk.base import SKClassifier, SKConfig
from _datasets.base import TLDataset
from sklearn.dummy import DummyClassifier
from typing import Dict, TypeVar
from _utils.enumerations import DataSplitEnum
from time import sleep


TDummyClf = TypeVar("TDummyClf", bound="DummyClf")


class DummyClfConfig(SKConfig):
    sleep: float


class DummyClf(SKClassifier):

    classifier: DummyClassifier
    config: DummyClfConfig

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, DummyClassifier, **kwargs)

        self.config = DummyClfConfig(**config)

    def fit(self, tl_dataset: TLDataset) -> TDummyClf:
        x_source, y_source = self.transform_dataset_to_X_Y_numpy(
            tl_dataset.source[DataSplitEnum.train]
        )

        self.classifier.fit(x_source, y_source)

        print("Fitted dummy classifier")

        print(f"Sleeping for {self.config.sleep} sec ...")
        sleep(self.config.sleep)
        print("Finished sleeping.")

        return self
