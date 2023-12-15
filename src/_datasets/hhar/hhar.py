"""Heterogeneity Activity Recognition Data Set."""

from enum import Enum
import numpy as np
import datasets
import os
import shutil
import glob
import math
import pandas as pd
from typing import List, Dict
from _utils.enumerations import DatasetColumnsEnum
from sklearn.model_selection import train_test_split


_CITATION = """\
@inproceedings{stisen2015smart,
  title={Smart devices are different: Assessing and mitigatingmobile sensing heterogeneities for activity recognition},
  author={Stisen, Allan and Blunck, Henrik and Bhattacharya, Sourav and Prentow, Thor Siiger and Kj{\ae}rgaard, Mikkel Baun and Dey, Anind and Sonne, Tobias and Jensen, Mads M{\o}ller},
  booktitle={Proceedings of the 13th ACM conference on embedded networked sensor systems},
  pages={127--140},
  year={2015}
}
"""

DATASET_NAME = "HHAR"

_DESCRIPTION = """\
    The Heterogeneity Dataset for Human Activity Recognition from Smartphone and Smartwatch sensors consists of two datasets devised to investigate sensor heterogeneities' impacts on human activity recognition algorithms (classification, automatic data segmentation, sensor fusion, feature extraction, etc).
"""

_HOMEPAGE = "http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition"

_LICENSE = "N/A"

_DOWNLOAD_LINK = "http://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip"

SEPERATOR = "."

TS_LEN = 128


class SubjectsEnum(str, Enum):
    a: str = "a"
    b: str = "b"
    c: str = "c"
    d: str = "d"
    e: str = "e"
    f: str = "f"
    g: str = "g"
    h: str = "h"
    i: str = "i"


CLASSES = ["stand", "sit", "walk", "stairsup", "stairsdown", "bike"]


class HHARConfig(datasets.BuilderConfig):
    """BuilderConfig for HHAR."""

    name: str
    n_splits: int
    test_size: float
    include: str
    exclude: str
    sampling: str

    def __init__(
        self,
        name: str,
        test_size=0.2,
        include: str = None,
        exclude: str = None,
        sampling: str = "25ms",
        **kwargs,
    ):
        """BuilderConfig for HHAR.

        Args:
            name: str # any custom name of this source/target based on SubjectsEnum
            include: str # this is a dot seperated string of SubjectsEnum
                values included in this dataset (eg "p1.p2")
                default:(None) means all subjects except the one(s) in `exclude`
            exclude: str # this is a dot seperated string of SubjectsEnum
                values excluded from this dataset (eg "p1.p2")
                default:(None) means all subjects except the one(s) in `include`
            test_size: the size of test set in percentage, split based on segment ID.
            sampling: the re-sampling frequency to have all series same time axis for example '50ms'
            **kwargs: keyword arguments forwarded to super.
        """
        super(HHARConfig, self).__init__(**kwargs)

        if include is None and exclude is None:
            raise Exception("include and exclude params cannot be both None")

        if include is not None and exclude is not None:
            raise Exception("include and exclude params cannot be both Not None")

        self.name = name
        self.n_splits = 1  # fixed
        self.test_size = test_size
        self.sampling = sampling

        self.exclude = exclude

        if include is not None:
            self.include = include
        else:
            # fill include by removing subjects mentioned in exclude
            self.include = f"{SEPERATOR}".join(
                [subject for subject in SubjectsEnum if subject not in self.exclude]
            )


class HHAR(datasets.GeneratorBasedBuilder):
    """Heterogeneity Activity Recognition Data Set."""

    BUILDER_CONFIG_CLASS = HHARConfig

    def _info(self):
        features = datasets.Features(
            {
                DatasetColumnsEnum.labels: datasets.features.ClassLabel(names=CLASSES),
                DatasetColumnsEnum.mts.value: [[datasets.Value("double")]],
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

        # download
        dataset_dir = f"/tmp/{DATASET_NAME}"

        # download if not already downloaded
        if not os.path.exists(dataset_dir):
            downloaded_dir = dl_manager.download_and_extract(_DOWNLOAD_LINK)
            shutil.copytree(downloaded_dir, dataset_dir)

        # allowed subjects as set
        allowed_subjects = set(
            [SubjectsEnum(subject).value for subject in self.config.include.split(SEPERATOR)]
        )

        data_dir = f"{dataset_dir}/Activity recognition exp"

        # min samples per user device gt
        min_samples = math.ceil(1 / self.config.test_size)

        # read phones acc data
        df = pd.read_csv(f"{data_dir}/Phones_accelerometer.csv")

        # filter users
        df = df[df["User"].isin(allowed_subjects)]

        # date time
        df["Arrival_Time"] = pd.to_datetime(df["Arrival_Time"], unit="ms")

        train_data = []
        test_data = []

        # loop through each (user, device. label)
        for k, g in df.groupby(["User", "Device", "gt"]):
            # init
            cur_data = []
            # get label of series
            label = g["gt"].values[0]
            # skip if label is null
            if pd.isna(label):
                continue
            # sort values by time
            g = g.sort_values(by=["Arrival_Time"])
            # set index for time
            g.index = g["Arrival_Time"]
            # resample frequency
            if self.config.sampling is not None:
                g = g.resample(self.config.sampling).median(numeric_only=True)
            # reset index
            g = g.reset_index(drop=True)
            # get idx of series
            idx_s = np.arange(TS_LEN * (g.shape[0] // TS_LEN)).reshape(-1, TS_LEN)
            # loop through each series
            segment_id = 0
            for idx in idx_s:
                # get the mts
                mts = g.loc[idx][["x", "y", "z"]]
                # skip if the whole series is NaN on any dimensions
                if mts.isna().values.all(axis=0).any():
                    continue
                # append it
                cur_data.append({"X": mts.values, "Y": label, "segment_id": segment_id})
                segment_id = segment_id + 1

            # skip if less than min samples
            if len(cur_data) < min_samples:
                continue

            # split into train test without shuffling
            # since cur_data is already sorted by idx (segment_id)
            train_size = int(len(cur_data) * (1 - self.config.test_size))

            # append this set of (user, device. label)
            # add to train dataset
            train_data.extend(cur_data[:train_size])
            # add to test dataset
            test_data.extend(cur_data[train_size:])

        # sort by segment_id to have train sorted by time and later split validation without shuffle
        train_data = sorted(train_data, key=lambda e: e["segment_id"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data": train_data},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data": test_data},
            ),
        ]

    def _generate_examples(self, data: List[dict]):
        """Yields examples."""
        # iterate the data
        for idx, row in enumerate(data):
            yield (
                idx,
                {
                    DatasetColumnsEnum.mts: row["X"].transpose().tolist(),
                    DatasetColumnsEnum.labels: row["Y"],
                },
            )
