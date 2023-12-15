"""Machine Fault Diagnosis"""

from enum import Enum
import datasets
from torch import load as torchLoad
from torch.utils.data import Dataset as torchDataset
from typing import List
from _utils.enumerations import DatasetColumnsEnum


_CITATION = """\
    @inproceedings{lessmeier2016condition,
    title={Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: A benchmark data set for data-driven classification},
    author={Lessmeier, Christian and Kimotho, James Kuria and Zimmer, Detmar and Sextro, Walter},
    booktitle={PHM Society European Conference},
    volume={3},
    number={1},
    year={2016}
    }
"""

DATASET_NAME = "MFD"

_DESCRIPTION = """\
    https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf
"""

_HOMEPAGE = "https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download"

_LICENSE = "https://creativecommons.org/licenses/by-nc/4.0/"

_DOWNLOAD_LINK = "https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/PU85XN"

SEPERATOR = "."

NUM_CHANNELS = 1

NUM_CONDITIONS = 4

ConditionsEnum = Enum("ConditionsEnum", {f"{i}": f"{i}" for i in range(0, NUM_CONDITIONS)})

TS_LEN = 5120

CLASSES = ["healthy", "inner-bearing", "outer-bearing"]


class MFDConfig(datasets.BuilderConfig):
    """BuilderConfig for MFD."""

    name: str
    include: str
    exclude: str
    pth: str

    def __init__(
        self,
        name: str,
        include: str = None,
        exclude: str = None,
        pth: str = f"/proj/tl-for-ts/data/{DATASET_NAME}/FD",
        **kwargs,
    ):
        """BuilderConfig for MFD.

        Args:
            name: str # any custom name of this source/target based on ConditionsEnum
            include: str # this is a dot seperated string of ConditionsEnum
                values included in this dataset (eg "p1.p2")
                default:(None) means all subjects except the one(s) in `exclude`
            exclude: str # this is a dot seperated string of ConditionsEnum
                values excluded from this dataset (eg "p1.p2")
                default:(None) means all subjects except the one(s) in `include`
            pth: path to the folder containing the data already splitted from {_DOWNLOAD_LINK}
            **kwargs: keyword arguments forwarded to super.
        """
        super(MFDConfig, self).__init__(**kwargs)

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
                [subject for subject in ConditionsEnum if subject not in self.exclude]
            )

        self.pth = pth


class MFD(datasets.GeneratorBasedBuilder):
    """MFD Data Set."""

    BUILDER_CONFIG_CLASS = MFDConfig

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

        # allowed conditions as set
        allowed_conditions = set(
            [ConditionsEnum(condtion).value for condtion in self.config.include.split(SEPERATOR)]
        )

        train_dataset = []
        test_dataset = []

        for condition in allowed_conditions:
            train_dataset.append(torchLoad(f"{self.config.pth}/train_{condition}.pt"))
            test_dataset.append(torchLoad(f"{self.config.pth}/test_{condition}.pt"))

        print("allowed_conditions", allowed_conditions)

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
        for condition in range(len(dataset)):
            n_train = dataset[condition]["samples"].shape[0]
            for i in range(n_train):
                mts = dataset[condition]["samples"][i, :]
                mts = mts[None, :]  # add a new dimension to indicate a single channel
                mts = mts.tolist()
                label = dataset[condition]["labels"][i].item()
                idx += 1
                yield (idx, {DatasetColumnsEnum.mts: mts, DatasetColumnsEnum.labels: label})
