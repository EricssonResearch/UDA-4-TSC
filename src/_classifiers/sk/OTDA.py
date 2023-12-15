from _classifiers.sk.base import SKClassifier, SKConfig
from _datasets.base import TLDataset
from sklearn.ensemble import RandomForestClassifier
import ot
from typing import Dict, TypeVar
import pickle as pkl
from typing import Tuple
import numpy as np
from _utils.enumerations import DataSplitEnum


TOTDA = TypeVar("TOTDA", bound="OTDA")


class OTDAConfig(SKConfig):
    adaptation: Dict


class OTDA(SKClassifier):
    """
    Optimal Transport for Domain Adaptation, based on the paper:
    Domain adaptation with regularized optimal transport. 2014. Courty et al.
    """

    adaptation: ot.da.SinkhornLpl1Transport
    classifier: RandomForestClassifier
    adaptation_fname: str = "adaptation.pkl"

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, RandomForestClassifier, **kwargs)
        # make sure config respects OTDA expected one
        self.config = OTDAConfig(**config)
        self.adaptation = ot.da.SinkhornLpl1Transport(**self.config.adaptation)

    def save_model(self) -> None:
        super().save_model()

        # dump the file in the train dir
        pkl_file = f"{self.train_dir}/{self.adaptation_fname}"
        with open(pkl_file, "wb") as f:
            pkl.dump(self.adaptation, f)

    def load_model(self, **args) -> None:
        super().load_model()
        pkl_file = f"{self.train_dir}/{self.adaptation_fname}"
        with open(pkl_file, "rb") as f:
            self.adaptation = pkl.load(f)

    def fit(self, tl_dataset: TLDataset) -> TOTDA:
        """
        Fit and transform the data using OTDA then fit a random forest for the classifier.
        """

        # retrieve the source and target train dataset
        x_source, y_source = self.transform_dataset_to_X_Y_numpy(
            tl_dataset.source[DataSplitEnum.train]
        )
        x_target, _ = self.transform_dataset_to_X_Y_numpy(tl_dataset.target[DataSplitEnum.train])

        # fit the OT formulation with the data
        self.adaptation.fit(x_source, y_source, x_target)

        # adapt the source data with the learned transport plan.
        x_source_adapted = self.adaptation.transform(x_source)

        # fit the classifier, here a random forest
        self.classifier.fit(x_source_adapted, y_source)

        print("Fitted OTDA")

        return self

    def pre_predict_source(
        self, x_source: np.array, y_source: np.array
    ) -> Tuple[np.array, np.array]:
        return self.adaptation.transform(x_source), y_source
