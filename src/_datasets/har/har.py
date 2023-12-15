"""Human Activity Recognition Using Smartphones Data Set."""

from enum import Enum
import numpy as np
import datasets
import os
import shutil
import glob
import pandas as pd
from typing import List, Dict
from _utils.enumerations import DatasetColumnsEnum
from sklearn.model_selection import train_test_split


_CITATION = """\
@inproceedings{anguita2013public,
  title={A public domain dataset for human activity recognition using smartphones.},
  author={Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge Luis and others},
  booktitle={Esann},
  volume={3},
  pages={3},
  year={2013}
}
"""

DATASET_NAME = "HAR"

_DESCRIPTION = """\
    The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70 percent of the volunteers was selected for generating the training data and 30 percent the test data.
"""

_HOMEPAGE = "https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones"

_LICENSE = "N/A"

_DOWNLOAD_LINK = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
)

SEPERATOR = "."

NUM_SUBJECTS = 30

# In some paper they only use acc_z, acc_y, and acc_z.
# In CoDats I've hust checked it's used with this 6 feature; but let's keep it in mind.
class FeatureNames(str, Enum):
    body_acc_x: str = "body_acc_x"
    body_acc_y: str = "body_acc_y"
    body_acc_z: str = "body_acc_z"
    body_gyro_x: str = "body_gyro_x"
    body_gyro_y: str = "body_gyro_y"
    body_gyro_z: str = "body_gyro_z"
    total_acc_x: str = "total_acc_x"
    total_acc_y: str = "total_acc_y"
    total_acc_z: str = "total_acc_z"


SubjectsEnum = Enum("SubjectsEnum", {f"{i}": f"{i}" for i in range(1, NUM_SUBJECTS + 1)})

NUM_LABELS = 6

TS_LEN = 128

CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


class HARConfig(datasets.BuilderConfig):
    """BuilderConfig for HAR."""

    name: str
    n_splits: int
    random_state: int
    test_size: float
    include: str
    exclude: str

    def __init__(
        self,
        name: str,
        test_size=0.3,
        include: str = None,
        exclude: str = None,
        random_state: int = 1,
        **kwargs,
    ):
        """BuilderConfig for HAR.

        Args:
            name: str # any custom name of this source/target based on SubjectsEnum
            include: str # this is a dot seperated string of SubjectsEnum
                values included in this dataset (eg "p1.p2")
                default:(None) means all subjects except the one(s) in `exclude`
            exclude: str # this is a dot seperated string of SubjectsEnum
                values excluded from this dataset (eg "p1.p2")
                default:(None) means all subjects except the one(s) in `include`
            test_size: the size of test set in percentage, split based on segment ID.
            random_state: the random state for splitting train test
            **kwargs: keyword arguments forwarded to super.
        """
        super(HARConfig, self).__init__(**kwargs)

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

        self.random_state = random_state


class HAR(datasets.GeneratorBasedBuilder):
    """HAR Data Set."""

    BUILDER_CONFIG_CLASS = HARConfig

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

        # read the dataframe
        df = []
        # loop through train and test
        for sp in ["train", "test"]:
            # init the df
            df_sp = pd.read_csv(
                f"{dataset_dir}/UCI HAR Dataset/{sp}/subject_{sp}.txt",
                header=None,
                names=["subject"],
                dtype={"subject": str},
                delim_whitespace=True,
            )
            # read each dimension of the mts
            for dim in FeatureNames:
                # filename
                fname = f"{dataset_dir}/UCI HAR Dataset/{sp}/Inertial Signals/{dim.value}_{sp}.txt"
                # read df for this dim
                dim_df = pd.read_csv(
                    fname,
                    delim_whitespace=True,
                    header=None,
                    names=[f"{dim}-{i}" for i in range(TS_LEN)],
                )
                # join it with old df
                df_sp = df_sp.join(dim_df)

            # read the labels
            df_labels = pd.read_csv(
                f"{dataset_dir}/UCI HAR Dataset/{sp}/y_{sp}.txt",
                delim_whitespace=True,
                header=None,
                names=["label"],
            )

            # join labels
            df_sp = df_sp.join(df_labels)

            df.append(df_sp)

        # concat data
        df = pd.concat(df, axis=0).reset_index(drop=True)
        # filter subjects
        df = df[df["subject"].isin(allowed_subjects)]

        # drop subject column
        df = df.drop("subject", axis=1)

        x_columns = list(df.columns)
        x_columns.remove("label")
        X = df[x_columns]
        Y = df["label"]

        # split train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, random_state=self.config.random_state, test_size=self.config.test_size, stratify=Y
        )

        # reset index
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"X": X_train, "y": y_train},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"X": X_test, "y": y_test},
            ),
        ]

    def _generate_examples(self, X: pd.DataFrame, y: pd.DataFrame):
        """Yields examples."""
        # iterate the data
        for idx in range(X.shape[0]):
            label = y.loc[idx] - 1
            mts = []
            for dim in FeatureNames:
                dim_cols = [f"{dim}-{i}" for i in range(TS_LEN)]
                mts.append(X.loc[idx, dim_cols].values.tolist())
            yield (
                idx,
                {
                    DatasetColumnsEnum.mts: mts,
                    DatasetColumnsEnum.labels: label,
                },
            )
