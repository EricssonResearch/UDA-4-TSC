"""PTB XL ECG Dataset."""

from enum import Enum
import datasets
from typing import List
from _utils.enumerations import DatasetColumnsEnum
import ast
import pandas as pd
from tqdm import tqdm
import wfdb
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split


# https://www.kaggle.com/code/khyeh0719/ptb-xl-dataset-wrangling/notebook

_CITATION = """\
@article{wagner2020ptb,
  title={PTB-XL, a large publicly available electrocardiography dataset},
  author={Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and Kreiseler, Dieter and Lunze, Fatima I and Samek, Wojciech and Schaeffter, Tobias},
  journal={Scientific data},
  volume={7},
  number={1},
  pages={154},
  year={2020},
  publisher={Nature Publishing Group UK London}
}
"""

DATASET_NAME = "ptbXLecg"

_DESCRIPTION = """New in PhysioBank is the PTB Diagnostic ECG Database, a collection of 549 high-resolution 15-lead ECGs (12 standard leads together with Frank XYZ leads), including clinical summaries for each record. From one to five ECG records are available for each of the 294 subjects, who include healthy subjects as well as patients with a variety of heart diseases."""

_HOMEPAGE = "https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset"

_LICENSE = "https://creativecommons.org/licenses/by/4.0/"

_DOWNLOAD_LINK = "https://storage.googleapis.com/kaggle-data-sets/1136210/1905968/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230505%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230505T145341Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=0b193cdf569e17ee7e42809653c40937275d921256016e5eeb024878da3bad37524d63ddad0faaa9bfa0f5d0dd08775fbce35093e36b91707d637cf11ab9997cf23c988e70e3bdf1d5c519f5a819c2bb23a1332fcfb40c2f85439fa1deca2897372d5e86a525399fd35828e0bfd16463e270c1a020a3fd35486db138d80c700f8ebcbb10e9c9469108c1f1da7153610a9802da8f50aaa9c669eea14080e48746c134177b3078ab0577e5909780da6c9227579ff2ec7ece718ccea805c3d0e9ae595c1c4aee8b0acfe1b81f7a2b13f41f9b3ece89f124c853ab49344116db54911043347a1e7c04268e0183b64bc762a39596418087e196eeeb5b8361cb9fbdac"

SEPERATOR = "."

ROOT_DOWNLOADS = "/proj/tl-for-ts/data"

NUM_DOMAINS = 4

NUM_CHANNELS = 12

DomainsEnum = Enum("DomainsEnum", {f"{i}": f"{i}" for i in range(NUM_DOMAINS)})

DOMAIN_TAG = "site"

NUM_LABELS = 5

TS_LEN = 1000

CLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]

SUB_PTH = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

CSV_FNAME = "ptbxl_database.csv"

LBL_COL = "diagnostic_superclass"


class ECGConfig(datasets.BuilderConfig):
    """BuilderConfig for ECG."""

    name: str
    include: str
    exclude: str
    pth: str

    def __init__(
        self,
        name: str,
        include: str = None,
        exclude: str = None,
        samp_rate: int = 100,
        nrows: int = None,
        test_size: float = 0.2,
        random_state: int = 1,
        **kwargs,
    ):
        f"""BuilderConfig for ECG dataset.

        Args:
            name: str # any custom name of this source/target based on DomainsEnum
            include: str # this is a dot seperated string of DomainsEnum
                values included in this dataset (eg "1.2")
                default:(None) means all subjects except the one(s) in `exclude`
            exclude: str # this is a dot seperated string of DomainsEnum
                values excluded from this dataset (eg "1.2")
                default:(None) means all subjects except the one(s) in `include`
            samp_rate: int # this is the sampling rate either 100 or 500
            nrows: int # the number of examples (None means all examples will be loaded)
            test_size: float # the size of test set
            random_state: int # random state to split the test set
            **kwargs: keyword arguments forwarded to super.
        """
        super(ECGConfig, self).__init__(**kwargs)

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
                [subject for subject in DomainsEnum if subject not in self.exclude]
            )

        self.nrows = nrows
        self.samp_rate = samp_rate
        self.random_state = random_state
        self.test_size = test_size


class PTBXLECG(datasets.GeneratorBasedBuilder):
    """PTBXLECG Data Set."""

    BUILDER_CONFIG_CLASS = ECGConfig

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

        dataset_dir = f"{ROOT_DOWNLOADS}/{DATASET_NAME}"

        # download if not already downloaded
        if not os.path.exists(dataset_dir):
            downloaded_dir = dl_manager.download_and_extract(_DOWNLOAD_LINK)
            shutil.copytree(downloaded_dir, dataset_dir)

        # allowed domains as set
        allowed_domains = [
            DomainsEnum(domain).value for domain in self.config.include.split(SEPERATOR)
        ]

        pth = f"{dataset_dir}/{SUB_PTH}"

        df = pd.read_csv(f"{pth}/{CSV_FNAME}", index_col="ecg_id", nrows=self.config.nrows)
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(pth + "/scp_statements.csv", index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_supclass_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        # Apply diagnostic superclass
        df[LBL_COL] = df.scp_codes.apply(aggregate_supclass_diagnostic)
        lbl_col_len = f"{LBL_COL}_len"
        df[lbl_col_len] = df[LBL_COL].apply(len)
        # keep only items with one class
        df = df.loc[df[lbl_col_len] == 1].reset_index(drop=True)

        # keep only allowed domains
        df = df[
            df[DOMAIN_TAG].fillna(NUM_DOMAINS).astype(int).astype(str).isin(allowed_domains)
        ].reset_index(drop=True)

        # load the mts data
        def load_raw_data(fname: str) -> np.ndarray:
            return np.array(wfdb.rdsamp(f"{pth}/{fname}")[0])

        if self.config.samp_rate == 100:
            fname_col = "filename_lr"
        elif self.config.samp_rate == 500:
            fname_col = "filename_hr"
        else:
            raise Exception("Supported sampling rate only 100 or 500")

        df[DatasetColumnsEnum.mts.value] = df[fname_col].apply(load_raw_data)

        print("df.shape:", df.shape)

        # split train test
        df_train, df_test = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df[LBL_COL].apply(lambda x: x[0]),
        )

        print("df_train.shape:", df_train.shape)
        print("df_test.shape:", df_test.shape)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"df": df_train},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"df": df_test},
            ),
        ]

    def _generate_examples(self, df: pd.DataFrame):
        """Yields examples."""
        for i in range(df.shape[0]):
            yield i, {
                DatasetColumnsEnum.labels: df.iloc[i][LBL_COL][0],
                DatasetColumnsEnum.mts: df.iloc[i][DatasetColumnsEnum.mts.value]
                .transpose()
                .tolist(),
            }
