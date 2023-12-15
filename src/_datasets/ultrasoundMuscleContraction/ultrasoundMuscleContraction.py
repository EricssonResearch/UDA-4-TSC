# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""1DUltrasoundMuscleContractionData."""


import os
from enum import Enum
import datasets
from typing import Dict, List
from collections import defaultdict
import arff
import shutil
import numpy as np
import math
from sklearn.model_selection import train_test_split
from _utils.enumerations import DatasetColumnsEnum


DATASET_NAME = "1DUltrasoundMuscleContractionData"

NUM_CLASSES = 2

ARFF_FILE_NAME = "file.arff"
ARFF_FILE_NAME_SMALL = "file.arff.small"
SEPERATOR = "."

_CITATION = """\
@article{brausch2022classifying,
  title={Classifying Muscle States with One-Dimensional Radio-Frequency Signals from Single Element Ultrasound Transducers},
  author={Brausch, Lukas and Hewener, Holger and Lukowicz, Paul},
  journal={Sensors},
  volume={22},
  number={7},
  pages={2789},
  year={2022},
  publisher={MDPI}
}
"""

_DESCRIPTION = """\
This collection includes 21 data sets of one-dimensional ultrasound raw RF data (A-Scans) 
acquired from the calf muscles of 8 healthy volunteers. The subjects were asked to manually 
annotate the data to allow for muscle contraction classification tasks. Each line of the 
ARFF file contains an A-Scan consisting of 3000 amplitude values, an annotation (whether 
this A-Scan belongs to a contracted muscle (1) or not (0)), the ID of the subject and the
 ID of the data set.
"""

_HOMEPAGE = "https://www.openml.org/search?type=data&sort=runs&id=41971&status=active"

_LICENSE = "N/A"

_URLs = {
    "1DUltrasoundMuscleContractionData": "https://www.openml.org/data/download/21379331/completeDatabase.arff",
}


class UltrasoundMuscleContractionEnum(str, Enum):
    subject1: str = "subject1"
    subject2: str = "subject2"
    subject3: str = "subject3"
    subject4: str = "subject4"
    subject5: str = "subject5"
    subject6: str = "subject6"
    subject7: str = "subject7"
    subject8: str = "subject8"


all_subjects: str = f"{SEPERATOR}".join([subject for subject in UltrasoundMuscleContractionEnum])


class UltrasoundMuscleContractionConfig(datasets.BuilderConfig):
    """
    BuilderConfig for UltrasoundMuscleContraction.
    name: str # any custom name of this source/target based on UltrasoundMuscleContraction
    include: str # this is a dot seperated string of UltrasoundMuscleContractionEnum
        values included in this dataset (eg "subject1.subject2")
        default:(None) means all subjects except the one(s) in `exclude`
    exclude: str # this is a dot seperated string of UltrasoundMuscleContractionEnum
        values excluded from this dataset (eg "subject1.subject2")
        default:(None) means all subjects except the one(s) in `include`
    random_state: random state that controls the split of train test
    test_size: the size of test set in percentage
    load_small: whether to load a small version for fast testing - works only with
        include and exclude either subject1 / subject8
    """

    name: str
    include: str
    exclude: str
    random_state: int
    test_size: float
    load_small: bool

    def __init__(
        self,
        name: str,
        include: str = None,
        exclude: str = None,
        random_state: int = 1,
        test_size: float = 0.2,
        load_small: bool = False,
        **kwargs,
    ):
        """BuilderConfig for UltrasoundMuscleContraction.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(UltrasoundMuscleContractionConfig, self).__init__(**kwargs)

        if include is None and exclude is None:
            raise Exception("include and exclude params cannot be both None")

        if include is not None and exclude is not None:
            raise Exception("include and exclude params cannot be both Not None")

        self.name = name
        self.random_state = random_state
        self.test_size = test_size
        self.load_small = load_small
        self.exclude = exclude

        if include is not None:
            self.include = include
        else:
            # fill include by removing subjects mentioned in exclude
            self.include = f"{SEPERATOR}".join(
                [
                    subject
                    for subject in UltrasoundMuscleContractionEnum
                    if subject not in self.exclude
                ]
            )


class UltrasoundMuscleContraction(datasets.GeneratorBasedBuilder):
    """UltrasoundMuscleContraction dataset."""

    BUILDER_CONFIG_CLASS = UltrasoundMuscleContractionConfig

    def _info(self):
        features = datasets.Features(
            {
                DatasetColumnsEnum.labels: datasets.features.ClassLabel(
                    names=[str(i) for i in range(NUM_CLASSES)]
                ),
                DatasetColumnsEnum.mts: [[datasets.Value("float")]],
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

        dataset_dir = f"/tmp/{DATASET_NAME}"
        out_file = f"{dataset_dir}/{ARFF_FILE_NAME}"
        # define small file in case we need it
        small_file = f"{dataset_dir}/{ARFF_FILE_NAME_SMALL}"

        # download if not already downloaded
        if not os.path.exists(out_file):
            my_urls = _URLs[DATASET_NAME]
            downloaded_dir = dl_manager.download_and_extract(my_urls)
            os.makedirs(dataset_dir, exist_ok=True)
            shutil.copy(downloaded_dir, out_file)
            # dump the smaller version in case loaded
            os.system(f"head -n 1000 {out_file} > {small_file}")
            os.system(f"tail -n 1000 {out_file} >> {small_file}")

        # check if load small
        if self.config.load_small is True:
            allowed_subjects_for_small = [
                UltrasoundMuscleContractionEnum.subject1.value,
                UltrasoundMuscleContractionEnum.subject8.value,
            ]
            # only works for subject1 / subject8
            assert (
                self.config.include in allowed_subjects_for_small
            ), "load_small==True only works for subject1 / subject8"
            out_file = small_file

        # load arff data
        print("loading arff......")
        data = arff.load(open(out_file, "r"))["data"]

        # allowed subjects as set
        allowed_subjects = set(
            [
                UltrasoundMuscleContractionEnum(subject).value.replace("subject", "")
                for subject in self.config.include.split(SEPERATOR)
            ]
        )

        X = []
        y = []

        print("process data depending on subjects")
        # loop over the rows in the dataset
        for row in data:
            # get the subject id
            subject_id = str(int(row[2]))
            # check if subject is allowed to be in
            if subject_id in allowed_subjects:
                # get the label
                label = str(int(row[1]))
                # extract the time series from str
                ts = [float(t) for t in row[0].replace("[", "").replace("]", "").split(",")]

                X.append(ts)
                y.append(label)

        del data

        # split train test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=self.config.random_state, test_size=self.config.test_size, stratify=y
        )

        print("Done splitting")

        del X
        del y

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

    def _generate_examples(self, X, y):
        """Yields examples."""

        # iterate the data
        for idx, row in enumerate(zip(X, y)):
            yield (idx, {DatasetColumnsEnum.mts: [row[0]], DatasetColumnsEnum.labels: row[1]})
