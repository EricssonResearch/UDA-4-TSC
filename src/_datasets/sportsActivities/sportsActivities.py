"""Daily and Sports Activities Data Set."""

from enum import Enum
import numpy as np
import datasets
import os
import shutil
import glob
import pandas as pd
from typing import List, Dict
from _utils.enumerations import DatasetColumnsEnum


_CITATION = """\
@article{altun2010comparative,
  title={Comparative study on classifying human activities with miniature inertial and magnetic sensors},
  author={Altun, Kerem and Barshan, Billur and Tun{\c{c}}el, Orkun},
  journal={Pattern Recognition},
  volume={43},
  number={10},
  pages={3605--3620},
  year={2010},
  publisher={Elsevier}
}
"""

DATASET_NAME = "SportsActivities"

_DESCRIPTION = """\
    The dataset comprises motion sensor data of 19 daily and sports activities each performed by 8 subjects in their own style for 5 minutes. Five Xsens MTx units are used on the torso, arms, and legs.
"""

_HOMEPAGE = "https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities#"

_LICENSE = "N/A"

_DOWNLOAD_LINK = "https://archive.ics.uci.edu/ml/machine-learning-databases/00256/data.zip"

SEPERATOR = "."


class SubjectsEnum(str, Enum):
    p1: str = "p1"
    p2: str = "p2"
    p3: str = "p3"
    p4: str = "p4"
    p5: str = "p5"
    p6: str = "p6"
    p7: str = "p7"
    p8: str = "p8"


LABELS = [f"a{str(i).zfill(2)}" for i in range(1, 20)]

NUM_SEGMENTS = 60

all_subjects: str = f"{SEPERATOR}".join([subject for subject in SubjectsEnum])


class SportsActivitiesConfig(datasets.BuilderConfig):
    """BuilderConfig for SportsActivities."""

    name: str
    n_splits: int
    test_size: float
    include: str
    exclude: str

    def __init__(
        self, name: str, test_size=0.2, include: str = None, exclude: str = None, **kwargs
    ):
        """BuilderConfig for SportsActivities.

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
        super(SportsActivitiesConfig, self).__init__(**kwargs)

        if include is None and exclude is None:
            raise Exception("include and exclude params cannot be both None")

        if include is not None and exclude is not None:
            raise Exception("include and exclude params cannot be both Not None")

        self.name = name
        self.n_splits = 1  # fixed
        self.test_size = test_size

        self.exclude = exclude

        if include is not None:
            self.include = include
        else:
            # fill include by removing subjects mentioned in exclude
            self.include = f"{SEPERATOR}".join(
                [subject for subject in SubjectsEnum if subject not in self.exclude]
            )


class SportsActivities(datasets.GeneratorBasedBuilder):
    """Daily and Sports Activities Data Set."""

    BUILDER_CONFIG_CLASS = SportsActivitiesConfig

    def _info(self):
        features = datasets.Features(
            {
                DatasetColumnsEnum.labels.value: datasets.ClassLabel(names=LABELS),
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

        # get list of all files while filtering allowed files
        files = [
            f
            for f in glob.glob(f"{dataset_dir}/data/a*/p*/s*.txt", recursive=True)
            if f.split("/")[-2] in allowed_subjects
        ]

        # get segment ID threshold based on test size
        segment_id_threshold = NUM_SEGMENTS - int(self.config.test_size * NUM_SEGMENTS)

        # def train and test lists
        train_set = []
        test_set = []

        # loop through files
        for fname in files:
            # read the data
            data = pd.read_csv(fname, header=None)
            # split based on '/'
            splits = fname.split("/")
            # get the label activity
            label = splits[-3]
            # get the segment id
            segment_id = int(splits[-1].split(".")[0].split("s")[1])
            # based on the segment threhsold decide if train or test
            if segment_id > segment_id_threshold:
                # then we go for test set
                test_set.append({"X": data, "y": label})
            else:
                # go for train set
                train_set.append({"X": data, "y": label, "segment_id": segment_id})

        train_set = sorted(train_set, key=lambda e: e["segment_id"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"xy_set": train_set},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"xy_set": test_set},
            ),
        ]

    def _generate_examples(self, xy_set: List[Dict]):
        """Yields examples."""
        # iterate the data
        for idx, row in enumerate(xy_set):
            yield (
                idx,
                {
                    DatasetColumnsEnum.mts: row["X"].to_numpy().transpose().tolist(),
                    DatasetColumnsEnum.labels: row["y"],
                },
            )
