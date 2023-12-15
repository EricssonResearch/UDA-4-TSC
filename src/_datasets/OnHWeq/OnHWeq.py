import os
import datasets
import pickle
import numpy as np
from enum import Enum
from sklearn.model_selection import train_test_split
from _utils.enumerations import DatasetColumnsEnum
from zipfile import ZipFile

_CITATION = """\
@inproceedings{ott_ijdar,
author = {Felix Ott and David Rügamer and Lucas Heublein and Tim Hamann and Jens Barth and Bernd Bischl and Christopher Mutschler},
title = {{Benchmarking Online Sequence-to-Sequence and Character-based Handwriting Recognition from IMU-Enhanced Pens}},
booktitle = {International Journal on Document Analysis and Recognition (IJDAR)},
month = sep,
year = {2022},
doi = {10.1007/s10032-022-00415-6}
}
"""

_DESCRIPTION = """\
Online Handwriting Recognition of equations from Sensor-Enhanced Pens.

To obtain the sensor data, a recording app that connects to a DigiPen and tells the volunteers what to write was implemented.

The OnHW-eq dataset does not contain or consider any sensor calibration.

OnHW-equations dataset was part of the UbiComp 2021 challenge1 and is written by 55 writers and consists of 10 number classes and 5 operator classes (+, -,·, :, =). The dataset consists of a total of 10,713 samples.

It is possible to split the sensor sequence based on the force sensor as the pen is lifted between every single character. This approach provides another useful dataset for a single character classification task. We set split constraints for long tip lifts and recursively split these sequences by assigning a possible number of strokes per character. This results in a total of 39,643 single characters.
"""

_HOMEPAGE = (
    "https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html"
)

_LICENSE = "N/A"

_URLs = {
    "OnHW_equations_R": "https://www2.iis.fraunhofer.de/LV-OnHW/OnHW-symbols_equations_indep.zip",
    "OnHW_equations_L": "https://www2.iis.fraunhofer.de/LV-OnHW/OnHW-symbols_equations_L.zip",
}


class OnHWDatasetNameEnum(str, Enum):
    OnHW_equations_R: str = "OnHW_equations_R"
    OnHW_equations_L: str = "OnHW_equations_L"


class OnHWeqConfig(datasets.BuilderConfig):
    """BuilderConfig for OnHWeq."""

    name: OnHWDatasetNameEnum
    test_size: float
    random_state: int

    def __init__(
        self, name: OnHWDatasetNameEnum, test_size: float = 0.3, random_state: int = 1, **kwargs
    ):
        """BuilderConfig for OnHW.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(OnHWeqConfig, self).__init__(**kwargs)
        self.name = OnHWDatasetNameEnum(name)
        self.test_size = test_size
        self.random_state = random_state


class OnHWeq(datasets.GeneratorBasedBuilder):
    """Online Handwriting Recognition from Sensor Enhanced Pens."""

    BUILDER_CONFIG_CLASS = OnHWeqConfig

    def _info(self):
        features = datasets.Features(
            {
                DatasetColumnsEnum.labels: datasets.Value("string"),
                DatasetColumnsEnum.mts: [[datasets.Value("double")]],
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
        print(self.config.name)

        """Returns SplitGenerators."""

        my_urls = _URLs[self.config.name.value]

        archive = dl_manager.download(my_urls)

        out_extracted = archive.split("downloads")[0]
        out_extracted = f"{out_extracted}{self.config.name.value}"
        done_out_extracted = f"{out_extracted}/DONE"

        if os.path.exists(done_out_extracted):
            print("Already extracted")
        else:
            with ZipFile(archive) as zf:
                zf.extractall(path=out_extracted)
            os.mkdir(out_extracted + "/DONE")

        if self.config.name == OnHWDatasetNameEnum.OnHW_equations_R:

            path_np = f"{out_extracted}/OnHW-symbols_equations_indep/"

            with open(path_np + "all_x_dat_train_imu_e.pkl", "rb") as f:
                X_train = pickle.load(f)
            with open(path_np + "all_x_dat_val_imu_e.pkl", "rb") as f:
                X_val = pickle.load(f)
            with open(path_np + "all_train_gt_e.pkl", "rb") as f:
                y_train = pickle.load(f)
            with open(path_np + "all_val_gt_e.pkl", "rb") as f:
                y_val = pickle.load(f)

            X = X_train + X_val
            y = y_train + y_val

            del X_train
            del X_val
            del y_train
            del y_val

        else:
            path_np = f"{out_extracted}/OnHW-symbols_equations_L/"

            with open(path_np + "all_x_dat_imu_e.pkl", "rb") as f:
                X = pickle.load(f)
            with open(path_np + "all_gt_e.pkl", "rb") as f:
                y = pickle.load(f)

        # Deleting Empty entries and Outliers

        empty_samples = []
        for i in range(len(X)):
            if len(X[i]) < 2:
                empty_samples.append(i)

        for i in empty_samples:
            del X[i]
            del y[i]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            random_state=self.config.random_state,
            test_size=self.config.test_size,
            stratify=y,
        )

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
        for id_ in range(len(X)):
            yield id_, {
                DatasetColumnsEnum.mts: X[id_].transpose().tolist(),
                DatasetColumnsEnum.labels: y[id_],
            }
