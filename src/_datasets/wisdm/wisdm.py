"""Activity Recognition using Cell Phone Accelerometers"""

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
@article{kwapisz2011activity,
  title={Activity recognition using cell phone accelerometers},
  author={Kwapisz, Jennifer R and Weiss, Gary M and Moore, Samuel A},
  journal={ACM SigKDD Explorations Newsletter},
  volume={12},
  number={2},
  pages={74--82},
  year={2011},
  publisher={ACM New York, NY, USA}
}
"""

DATASET_NAME = "WISDM"

_DESCRIPTION = """\
    https://www.cis.fordham.edu/wisdm/includes/files/sensorKDD-2010.pdf
"""

_HOMEPAGE = "https://www.cis.fordham.edu/wisdm/dataset.php"

_LICENSE = "N/A"

_DOWNLOAD_LINK = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"

SEPERATOR = "."

TS_LEN = 128

CLASSES = ["Jogging", "Walking", "Upstairs", "Downstairs", "Sitting", "Standing"]

subjects: List[int] = [
    33,
    17,
    20,
    29,
    13,
    15,
    6,
    27,
    36,
    18,
    32,
    35,
    11,
    16,
    5,
    10,
    28,
    26,
    14,
    24,
    12,
    23,
    4,
    30,
    34,
    8,
    31,
    21,
    3,
    22,
    1,
    9,
    25,
    2,
    7,
    19,
]

SubjectsEnum = Enum("SubjectsEnum", {f"{i}": f"{i}" for i in subjects})


class WISDMConfig(datasets.BuilderConfig):
    """BuilderConfig for WISDM."""

    name: str
    test_size: float
    include: str
    exclude: str

    def __init__(
        self,
        name: str,
        test_size=0.2,
        include: str = None,
        exclude: str = None,
        **kwargs,
    ):
        """BuilderConfig for WISDM.

        Args:
            name: str # any custom name of this source/target based on SubjectsEnum
            include: str # this is a dot seperated string of SubjectsEnum
                values included in this dataset (eg "p1.p2")
                default:(None) means all subjects except the one(s) in `exclude`
            exclude: str # this is a dot seperated string of SubjectsEnum
                values excluded from this dataset (eg "p1.p2")
                default:(None) means all subjects except the one(s) in `include`
            test_size: the size of test set in percentage, split based on segment ID.
            **kwargs: keyword arguments forwarded to super.
        """
        super(WISDMConfig, self).__init__(**kwargs)

        if include is None and exclude is None:
            raise Exception("include and exclude params cannot be both None")

        if include is not None and exclude is not None:
            raise Exception("include and exclude params cannot be both Not None")

        self.name = name
        self.test_size = test_size

        self.exclude = exclude

        if include is not None:
            self.include = include
        else:
            # fill include by removing subjects mentioned in exclude
            self.include = f"{SEPERATOR}".join(
                [subject for subject in SubjectsEnum if subject not in self.exclude]
            )


class WISDM(datasets.GeneratorBasedBuilder):
    """Activity Recognition using Cell Phone Accelerometers."""

    BUILDER_CONFIG_CLASS = WISDMConfig

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

        fname = f"{dataset_dir}/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"

        df = pd.read_csv(
            fname,
            on_bad_lines="warn",
            names=["user", "label", "timestamp", "x", "y", "z", "todrop"],
            index_col=False,
        )

        df = df.drop("todrop", axis=1)

        df["z"] = (
            df["z"].apply(lambda x: x.split(";")[0] if isinstance(x, str) else x).astype("float32")
        )

        # allowed subjects as set
        allowed_subjects = set(
            [SubjectsEnum(subject).value for subject in self.config.include.split(SEPERATOR)]
        )

        # user col to str
        df["user"] = df["user"].astype(str)

        # filter users
        df = df[df["user"].isin(allowed_subjects)].reset_index(drop=True)

        # date time
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")

        # min samples per user , label
        min_samples = math.ceil(1 / self.config.test_size)

        train_data = []
        test_data = []

        # loop through each (user, label)
        for k, g in df.groupby(["user", "label"]):
            # init
            cur_data = []
            # get label of series
            label = k[1]
            # skip if label is null
            if pd.isna(label):
                raise Exception("label null for user", k)
            # sort values by time
            g = g.sort_values(by=["timestamp"])

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
                raise Exception(f"min_samples {min_samples} not met for ", k)

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
