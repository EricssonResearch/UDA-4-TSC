from _preprocessing.Preprocessor import Preprocessor
from datasets import Dataset, ClassLabel
from typing import TypeVar, Dict
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder
import pandas as pd
from _utils.enumerations import DatasetColumnsEnum


TLabelEncoder = TypeVar("TLabelEncoder", bound="LabelEncoder")


class LabelEncoder(Preprocessor):

    verbose: bool
    model: SKLabelEncoder = SKLabelEncoder()

    class Config:
        arbitrary_types_allowed = True

    def fit(self, X: Dataset) -> TLabelEncoder:
        if self.verbose is True:
            print("Fitting LabelEncoder ...")
            df = pd.DataFrame(X)
            self.model.fit(df[DatasetColumnsEnum.labels])
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        # define a function that maps a single example in the dataset
        def map_function(example: Dict) -> Dict:
            # transform the label
            example[DatasetColumnsEnum.labels] = self.model.transform(
                [example[DatasetColumnsEnum.labels]]
            )[0]
            return example

        # create new features *deep by default
        features = dataset.features.copy()
        # reset the column 'label' to be `ClassLabel` in huggingface
        features[DatasetColumnsEnum.labels] = ClassLabel(names=self.model.classes_.tolist())
        # apply the mapping function on the whole dataset
        dataset = dataset.map(map_function, features=features)
        # return the new dataset
        return dataset
