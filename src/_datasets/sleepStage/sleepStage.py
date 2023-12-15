"""Sleep Stage EEG Classification Dataset."""

from enum import Enum
import datasets
from torch import load as torchLoad
from torch.utils.data import Dataset as torchDataset
from typing import List
from _utils.enumerations import DatasetColumnsEnum


_CITATION = """\
@data{N9/UD1IM9_2022,
author = {Ragab, Mohamed and Eldele, Emadeldeen},
publisher = {DR-NTU (Data)},
title = {{Subject-wise Sleep Stage Data}},
year = {2022},
version = {V1},
doi = {10.21979/N9/UD1IM9},
url = {https://doi.org/10.21979/N9/UD1IM9}
}
"""

DATASET_NAME = "sleepStage"

_DESCRIPTION = """Sleep stage classification (SSC) problem aims to classify the electroencephalography (EEG) signals into five stages i.e. Wake (W), Non-Rapid Eye Movement stages (N1, N2, N3), and Rapid Eye Movement (REM). We adopted Sleep-EDF dataset (Goldberger et al., 2000), which contains EEG readings from 20 healthy subjects (2022-03-06)"""

_HOMEPAGE = "https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9"

_LICENSE = "https://creativecommons.org/licenses/by-nc/4.0/"

_DOWNLOAD_LINK = "https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9"

SEPERATOR = "."

NUM_SUBJECTS = 20

NUM_CHANNELS = 1

SubjectsEnum = Enum("SubjectsEnum", {f"{i}": f"{i}" for i in range(0, NUM_SUBJECTS)})

NUM_LABELS = 5

TS_LEN = 3000

CLASSES = ["W", "N1", "N2", "N3", "REM"]


class SleepStageConfig(datasets.BuilderConfig):
    """BuilderConfig for SleepStage."""

    name: str
    include: str
    exclude: str
    pth: str

    def __init__(
        self,
        name: str,
        include: str = None,
        exclude: str = None,
        pth: str = f"/proj/tl-for-ts/data/{DATASET_NAME}/EEG",
        **kwargs,
    ):
        f"""BuilderConfig for HarCoTMix.

        Args:
            name: str # any custom name of this source/target based on SubjectsEnum
            include: str # this is a dot seperated string of SubjectsEnum
                values included in this dataset (eg "1.2")
                default:(None) means all subjects except the one(s) in `exclude`
            exclude: str # this is a dot seperated string of SubjectsEnum
                values excluded from this dataset (eg "1.2")
                default:(None) means all subjects except the one(s) in `include`
            pth: path to the folder containing the data already splitted from {_HOMEPAGE}
            **kwargs: keyword arguments forwarded to super.
        """
        super(SleepStageConfig, self).__init__(**kwargs)

        if include is None and exclude is None:
            raise Exception("include and exclude params cannot be both None")

        if include is not None and exclude is not None:
            raise Exception("include and exclude params cannot be both Not None")

        self.name = name

        self.exclude = exclude

        if include is not None:
            self.include = include
        else:
            # fill include by removing subjects mentioned in exclude
            self.include = f"{SEPERATOR}".join(
                [subject for subject in SubjectsEnum if subject not in self.exclude]
            )

        self.pth = pth


class SleepStage(datasets.GeneratorBasedBuilder):
    """Sleep Stage Data Set."""

    BUILDER_CONFIG_CLASS = SleepStageConfig

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

        # allowed subjects as set
        allowed_subjects = [
            SubjectsEnum(subject).value for subject in self.config.include.split(SEPERATOR)
        ]

        train_dataset = []
        test_dataset = []

        for subject in allowed_subjects:
            train_dataset.append(torchLoad(f"{self.config.pth}/train_{subject}.pt"))
            test_dataset.append(torchLoad(f"{self.config.pth}/test_{subject}.pt"))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"dataset": train_dataset},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"dataset": test_dataset},
            ),
        ]

    def _generate_examples(self, dataset: List[torchDataset]):
        """Yields examples."""
        idx = -1
        for subject in range(len(dataset)):
            n_train = dataset[subject]["samples"].shape[0]
            for i in range(n_train):
                mts = dataset[subject]["samples"][i, :, :].transpose().tolist()
                label = dataset[subject]["labels"][i]
                idx += 1
                yield (idx, {DatasetColumnsEnum.mts: mts, DatasetColumnsEnum.labels: label})
