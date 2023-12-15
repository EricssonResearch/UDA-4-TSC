from _preprocessing.Preprocessor import Preprocessor
from datasets import Dataset, ClassLabel, Array2D
from typing import TypeVar, Dict
import numpy as np
from _utils.enumerations import DatasetColumnsEnum


TToPyArrow = TypeVar("TToPyArrow", bound="ToPyArrow")


class ToPyArrow(Preprocessor):
    def fit(self, X: Dataset) -> TToPyArrow:
        print("Dummy fit for ToPyArrow")
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        print("Transforming to pyarrow arrays using ToPyArrow")
        # define a function that maps a single example in the dataset
        def map_function(example: Dict) -> Dict:
            # transform to numpy array
            example[DatasetColumnsEnum.mts] = np.array(example[DatasetColumnsEnum.mts])
            return example

        # get the shape of the mts based on first example
        first_example = dataset[DatasetColumnsEnum.mts][0]
        shape = (len(first_example), len(first_example[0]))
        # create new features *deep by default
        features = dataset.features.copy()
        # reset the column 'mts' to be `Array2D` in huggingface
        features[DatasetColumnsEnum.mts] = Array2D(shape=shape, dtype="float64")
        # set to numpy format
        dataset.set_format("numpy")
        # apply the mapping function on the whole dataset
        dataset = dataset.map(map_function, features=features)
        # return the new dataset
        return dataset
