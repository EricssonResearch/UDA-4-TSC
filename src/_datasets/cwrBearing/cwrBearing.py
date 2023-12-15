"""Case Western Reserve Bearing Data Set."""

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
import scipy

# code inspired from https://github.com/XiongMeijing/CWRU-1/blob/master/helper.py


_CITATION = """\
    @article{zhang2019machine,
  title={Machine learning and deep learning algorithms for bearing fault diagnostics-a comprehensive review. arXiv preprints},
  author={Zhang, Shen and Zhang, Shibo and Wang, Bingnan and Habetler, Thomas G},
  journal={arXiv preprint arXiv:1901.08247},
  year={2019}
    }
"""

DATASET_NAME = "cwrBearing"

_DESCRIPTION = """\
    Data was collected for normal bearings, single-point drive end and fan end defects.  Data was collected at 12,000 samples/second and at 48,000 samples/second for drive end bearing experiments.  All fan end bearing data was collected at 12,000 samples/second.
"""

_HOMEPAGE = "https://engineering.case.edu/bearingdatacenter"

_LICENSE = "N/A"

_DOWNLOAD_LINK = "https://github.com/XiongMeijing/CWRU-1/archive/refs/heads/master.zip"

SEPERATOR = "."


class ConditionsEnum(str, Enum):
    """Conditions for the motor"""

    condition_0: str = "0"
    condition_1: str = "1"
    condition_2: str = "2"
    condition_3: str = "3"


LABELS = ["Normal", "Ball", "InnerRaceway", "OuterRaceway"]

TS_LEN = 512


class CWRBearingConfig(datasets.BuilderConfig):
    """BuilderConfig for CWRBearing."""

    name: str
    test_size: float

    def __init__(self, name: str, test_size=0.2, **kwargs):
        """BuilderConfig for CWRBearing.

        Args:
            name: ConditionsEnum # the motor condition (domain)
            test_size: the size of test set in percentage, split based on segment ID.
            **kwargs: keyword arguments forwarded to super.
        """
        super(CWRBearingConfig, self).__init__(**kwargs)

        self.name = ConditionsEnum(name).value
        self.test_size = test_size


class CWRBearing(datasets.GeneratorBasedBuilder):
    """CWRBearing Data Set."""

    BUILDER_CONFIG_CLASS = CWRBearingConfig

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

        dataset_dir = f"{dataset_dir}/CWRU-1-master/Data/*/"

        data = []

        cols = ["DE_time"]

        for cur_dir in glob.glob(dataset_dir):

            df = matfile_to_df(Path(f"{cur_dir}"), self.config.name)

            for i in range(df.shape[0]):

                label = df.iloc[i]["label"]

                mts = np.asarray([df.iloc[i][col] for col in cols]).squeeze(-1)

                len_ts = mts.shape[1]

                idx_s = np.arange(TS_LEN * (len_ts // TS_LEN)).reshape(-1, TS_LEN)
                segment_id = 0
                for idx in idx_s:

                    x = mts[:, idx]
                    data.append({"x": x, "y": label, "segment_id": segment_id})
                    segment_id += 1

        data = sorted(data, key=lambda e: e["segment_id"])

        test_size = int(len(data) * self.config.test_size)

        train_size = len(data) - test_size

        train_data = data[:train_size]

        test_data = data[train_size:]

        # sort train dataset since will be helpful for splitting validation
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

    def _generate_examples(self, data: List[Dict]):
        """Yields examples."""
        # iterate the data
        for idx, row in enumerate(data):
            assert row["x"].shape[1] == TS_LEN
            assert row["y"] is not None
            yield (
                idx,
                {
                    DatasetColumnsEnum.mts: row["x"].tolist(),
                    DatasetColumnsEnum.labels: row["y"],
                },
            )


def matfile_to_dic(folder_path, condition):
    """
    Read all the matlab files of the CWRU Bearing Dataset and return a
    dictionary. The key of each item is the filename and the value is the data
    of one matlab file, which also has key value pairs.

    Parameter:
        folder_path:
            Path (Path object) of the folder which contains the matlab files.
    Return:
        output_dic:
            Dictionary which contains data of all files in the folder_path.
    """
    output_dic = {}
    for _, filepath in enumerate(folder_path.glob(f"*_{condition}.mat")):
        # strip the folder path and get the filename only.
        key_name = str(filepath).split("\\")[-1]
        output_dic[key_name] = scipy.io.loadmat(filepath)
    return output_dic


def remove_dic_items(dic):
    """
    Remove redundant data in the dictionary returned by matfile_to_dic inplace.
    """
    # For each file in the dictionary, delete the redundant key-value pairs
    for _, values in dic.items():
        del values["__header__"]
        del values["__version__"]
        del values["__globals__"]


def rename_keys(dic):
    """
    Rename some keys so that they can be loaded into a
    DataFrame with consistent column names
    """
    # For each file in the dictionary
    for _, v1 in dic.items():
        # For each key-value pair, rename the following keys
        for k2, _ in list(v1.items()):
            if "DE_time" in k2:
                v1["DE_time"] = v1.pop(k2)
            elif "BA_time" in k2:
                v1["BA_time"] = v1.pop(k2)
            elif "FE_time" in k2:
                v1["FE_time"] = v1.pop(k2)
            elif "RPM" in k2:
                v1["RPM"] = v1.pop(k2)


def label(filename):
    """
    Function to create label for each signal based on the filename. Apply this
    to the "filename" column of the DataFrame.
    Usage:
        df['label'] = df['filename'].apply(label)
    """
    if "/B" in filename:
        return "Ball"
    elif "/IR" in filename:
        return "InnerRaceway"
    elif "/OR" in filename:
        return "OuterRaceway"
    elif "/Normal" in filename:
        return "Normal"


def matfile_to_df(folder_path, condition):
    """
    Read all the matlab files in the folder, preprocess, and return a DataFrame

    Parameter:
        folder_path:
            Path (Path object) of the folder which contains the matlab files.
        condition:
            ConditionsEnum: the condition of the motor between 0 and 3
    Return:
        DataFrame with preprocessed data
    """
    dic = matfile_to_dic(folder_path, condition)
    remove_dic_items(dic)
    rename_keys(dic)
    df = pd.DataFrame.from_dict(dic).T
    df = df.reset_index().rename(mapper={"index": "filename"}, axis=1)
    df["label"] = df["filename"].apply(label)
    return df.drop(["RPM", "ans"], axis=1, errors="ignore")
