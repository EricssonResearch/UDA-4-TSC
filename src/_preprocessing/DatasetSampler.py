from _preprocessing.Preprocessor import Preprocessor
from datasets import Dataset
from typing import TypeVar, Union
from _utils.enumerations import DatasetColumnsEnum


TDatasetSampler = TypeVar("TDatasetSampler", bound="DatasetSampler")


class DatasetSampler(Preprocessor):

    sample_size: Union[float, int]
    stratify_by_column: str = DatasetColumnsEnum.labels
    seed: int = 1

    class Config:
        arbitrary_types_allowed = True

    def fit(self, X: Dataset) -> TDatasetSampler:
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        return dataset.train_test_split(
            test_size=self.sample_size,
            stratify_by_column=self.stratify_by_column,
            seed=self.seed,
        )["test"]
