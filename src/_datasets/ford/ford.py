"""The FordA/B Dataset for time series classification."""


import os
import csv
from enum import Enum
import numpy as np
import datasets
from typing import Dict
from sklearn.model_selection import StratifiedShuffleSplit
from _utils.enumerations import DatasetColumnsEnum


_CITATION = """\
@article{dau2019ucr,
  title={The UCR time series archive},
  author={Dau, Hoang Anh and Bagnall, Anthony and Kamgar, Kaveh and Yeh, Chin-Chia Michael and Zhu, Yan and Gharghabi, Shaghayegh and Ratanamahatana, Chotirat Ann and Keogh, Eamonn},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={6},
  number={6},
  pages={1293--1305},
  year={2019},
  publisher={IEEE}
}
"""

_DESCRIPTION = """\
    FordA: This data was originally used in a competition in the IEEE World Congress on Computational Intelligence, 2008. The classification problem is to diagnose whether a certain symptom exists or does not exist in an automotive subsystem. Each case consists of 500 measurements of engine noise and a classification. There are two separate problems: For FordA the Train and test data set were collected in typical operating conditions, with minimal noise contamination. 
    FordB: This data was originally used in a competition in the IEEE World Congress on Computational Intelligence, 2008. The classification problem is to diagnose whether a certain symptom exists or does not exist in an automotive subsystem. Each case consists of 500 measurements of engine noise and a classification. There are two separate problems: For FordB the training data were collected in typical operating conditions, but the test data samples were collected under noisy conditions.
"""

_HOMEPAGE = "http://timeseriesclassification.com/description.php?Dataset=FordA"
# Also: http://timeseriesclassification.com/description.php?Dataset=FordB

_LICENSE = "N/A"

UCR2018 = "_datasets/ucr2018"


class FordDatasetNameEnum:
    FordA: str = "FordA"
    FordB: str = "FordB"


class FordConfig(datasets.BuilderConfig):
    """BuilderConfig for FordDataset."""

    name: FordDatasetNameEnum
    password: str
    n_splits: int
    random_state: int
    test_size: float

    def __init__(
        self, name: FordDatasetNameEnum, password: str, random_state=1, test_size=0.2, **kwargs
    ):
        """BuilderConfig for FordDataset.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FordConfig, self).__init__(**kwargs)
        self.name = name
        self.password = password
        self.n_splits = 1  # fixed
        self.random_state = random_state
        self.test_size = test_size

        # password should be changed see the {_HOMEPAGE}


class FordDataset(datasets.GeneratorBasedBuilder):
    """UCR Time Series Archive 2018."""

    BUILDER_CONFIG_CLASS = FordConfig

    def _info(self):
        features = datasets.Features(
            {
                DatasetColumnsEnum.labels: datasets.Value("string"),
                DatasetColumnsEnum.mts: [[datasets.Value("double")]],
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        # load using the already defined ucr2018 class
        ford_dataset: datasets.DatasetDict = datasets.load_dataset(
            UCR2018, name=self.config.name, password=self.config.password
        )

        # split train test if FordB (see description why)
        if self.config.name == FordDatasetNameEnum.FordB:
            folds = StratifiedShuffleSplit(
                n_splits=self.config.n_splits,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            train_idx, test_idx = next(
                folds.split(
                    np.zeros(ford_dataset[datasets.Split.TEST].num_rows),
                    ford_dataset[datasets.Split.TEST][DatasetColumnsEnum.labels],
                )
            )
            ford_dataset = datasets.DatasetDict(
                {
                    datasets.Split.TRAIN: ford_dataset[datasets.Split.TEST].select(train_idx),
                    datasets.Split.TEST: ford_dataset[datasets.Split.TEST].select(test_idx),
                }
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"dataset": ford_dataset[datasets.Split.TRAIN]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"dataset": ford_dataset[datasets.Split.TEST]},
            ),
        ]

    def _generate_examples(self, dataset):
        """Yields examples."""
        for i in range(dataset.num_rows):
            yield i, dataset[i]
