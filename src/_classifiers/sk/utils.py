import numpy as np
from typing import Tuple
from datasets import Dataset
from _utils.enumerations import *


def transform_dataset_to_X_Y_numpy(dataset: Dataset) -> Tuple[np.array, np.array]:
    n_samples = len(dataset)
    X_col = DatasetColumnsEnum.mts.value
    if DatasetColumnsEnum.ori_mts in dataset.features.keys():
        X_col = DatasetColumnsEnum.ori_mts
    X = np.asarray(dataset[X_col]).reshape(n_samples, -1)
    Y = np.asarray(dataset[DatasetColumnsEnum.labels]).reshape(
        n_samples,
    )
    return X, Y
