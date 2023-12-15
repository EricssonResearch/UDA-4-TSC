"""miniTimeMatch Data Set."""

from enum import Enum
import numpy as np
import datasets
import os
import shutil
import glob
import pandas as pd
from typing import List, Dict
from _utils.enumerations import DatasetColumnsEnum
from pathlib import Path
import pickle as pkl
from sklearn.model_selection import train_test_split


_CITATION = """\

"""

DATASET_NAME = "miniTimeMatch"

_DESCRIPTION = """\
    .
"""

_HOMEPAGE = "https://volumen.univ-ubs.fr/rifzbqdc71"

_LICENSE = "N/A"

_DOWNLOAD_LINK = "https://volumen.univ-ubs.fr/rifzbqdc71/download"

SEPERATOR = "."


class ConditionsEnum(str, Enum):
    """Conditions for the motor"""

    FR1: str = "FR1"
    FR2: str = "FR2"
    DK1: str = "DK1"
    AT1: str = "AT1"


ALL_LABELS = [
    "corn",
    "horsebeans",
    "meadow",
    "spring barley",
    "spring oat",
    "spring peas",
    "spring rapeseed",
    "spring rye",
    "spring triticale",
    "spring wheat",
    "sunflowers",
    "unknown",
    "winter barley",
    "winter oat",
    "winter peas",
    "winter rapeseed",
    "winter rye",
    "winter triticale",
    "winter wheat",
]

LABELS_LIST = [
    "corn",
    "horsebeans",
    "meadow",
    "spring barley",
    "winter barley",
    "winter rapeseed",
    "winter triticale",
    "winter wheat",
]

LABELS = {ALL_LABELS.index(lbl): lbl for lbl in LABELS_LIST}

NUM_CHANNELS = 10


class MiniTimeMatchConfig(datasets.BuilderConfig):
    """BuilderConfig for MiniTimeMatch."""

    name: str
    test_size: float
    random_state: int

    def __init__(self, name: str, test_size: float = 0.2, random_state: int = 1, **kwargs):
        """BuilderConfig for MiniTimeMatch.

        Args:
            name: ConditionsEnum # the motor condition (domain)
            test_size: the size of test set in percentage, split based on segment ID.
            random_state: random state for splitting train test split in stratified manner
            **kwargs: keyword arguments forwarded to super.
        """
        super(MiniTimeMatchConfig, self).__init__(**kwargs)

        self.name = ConditionsEnum(name).value
        self.test_size = test_size
        self.random_state = random_state


class MiniTimeMatch(datasets.GeneratorBasedBuilder):
    """CWRBearing Data Set."""

    BUILDER_CONFIG_CLASS = MiniTimeMatchConfig

    def _info(self):
        features = datasets.Features(
            {
                DatasetColumnsEnum.labels.value: datasets.ClassLabel(names=LABELS_LIST),
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

        with open(f"{dataset_dir}/miniTimeMatch_datasets/miniTimeMatch", "rb") as f:
            ds = pkl.load(f)

        with open(f"{dataset_dir}/miniTimeMatch_datasets/miniTimeMatch_lab", "rb") as f:
            labels = pkl.load(f)

        domain_names = ["FR1", "FR2", "DK1", "AT1"]
        idx_domain = domain_names.index(self.config.name)

        ds = ds[idx_domain]
        labels = labels[idx_domain]

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            ds,
            labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=labels,
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data": X_train, "labels": y_train},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data": X_test, "labels": y_test},
            ),
        ]

    def _generate_examples(self, data: np.ndarray, labels: np.ndarray):
        """Yields examples."""
        # iterate the data
        for idx in range(data.shape[0]):
            mts = data[idx].transpose()
            label = LABELS[int(labels[idx])]
            assert mts.shape[0] == NUM_CHANNELS
            yield (
                idx,
                {
                    DatasetColumnsEnum.mts: mts.tolist(),
                    DatasetColumnsEnum.labels: label,
                },
            )
