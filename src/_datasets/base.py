from pydantic import BaseModel
from datasets import DatasetDict, load_from_disk, Dataset, ClassLabel
from typing import List, TypeVar, Dict
from _preprocessing.Preprocessor import Preprocessor
from _utils.enumerations import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import GaussianMixture
from _classifiers.sk.utils import transform_dataset_to_X_Y_numpy
import numpy as np
import time
import pandas as pd
from _utils.paths import Paths
import pickle as pkl
import shutil
import os


TTLDataset = TypeVar("TTLDataset", bound="TLDataset")


class TLDataset(BaseModel):

    dataset_name: DatasetNameEnum
    preprocessing_pipeline: List[Preprocessor] = None
    source: DatasetDict = None
    target: DatasetDict = None
    gmm_root_dir: str = None

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _fit_gmm(n_components: int, dataset: Dataset) -> GaussianMixture:
        """
        Estimate the density of the train part of the dataset using a GGM
        """

        print("Start estimate of GaussianMixture")
        start_time = time.time()

        # transform the data to numpy, keep only the feature to estimate the density
        X = np.asarray(dataset[DatasetColumnsEnum.mts.value]).reshape(len(dataset), -1)

        # compute the density
        GMM = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            init_params="k-means++",
            random_state=1,
        )

        res = GMM.fit(X)

        print(f"Finished fitting GaussianMixture, time took: {time.time()-start_time}")

        return res

    def fit_gmm(self, n_components: int) -> None:
        """
        Will fit a GaussianMixture model of n_components for source and target test sets
        and save the sklearn output model locally in a pickle file for each source target
        """
        self.gmm_root_dir = Paths.get_gmm_root_dir()
        source_gmm = self._fit_gmm(n_components, self.source[DataSplitEnum.test])
        fname = Paths.get_fname_gmm(
            gmm_root_dir=self.gmm_root_dir,
            domain_name=DomainNameEnum.source,
            n_components=n_components,
        )
        with open(fname, "wb") as f:
            pkl.dump(source_gmm, f)

        target_gmm = self._fit_gmm(n_components, self.target[DataSplitEnum.test])
        fname = Paths.get_fname_gmm(
            gmm_root_dir=self.gmm_root_dir,
            domain_name=DomainNameEnum.target,
            n_components=n_components,
        )
        with open(fname, "wb") as f:
            pkl.dump(target_gmm, f)

    def get_gmm(self, domain_name: DomainNameEnum, n_components: int) -> GaussianMixture:
        fname = Paths.get_fname_gmm(
            gmm_root_dir=self.gmm_root_dir, domain_name=domain_name, n_components=n_components
        )
        with open(fname, "rb") as f:
            return pkl.load(f)

    def get_density_at_source_point(self, n_components: int) -> Dict[DomainNameEnum, np.ndarray]:
        """Will take the fitted gmms and return the target/source density at source point"""
        # compute the two densities for the source and target data
        source_density = self.get_gmm(domain_name=DomainNameEnum.source, n_components=n_components)
        target_density = self.get_gmm(domain_name=DomainNameEnum.target, n_components=n_components)

        # Transform to numpy dataset
        X_source, _ = transform_dataset_to_X_Y_numpy(self.source[DataSplitEnum.test])

        # compute the value of the density at each source point
        source_density_at_source_point = (
            source_density.predict_proba(X_source) @ source_density.weights_
        )
        target_density_at_source_point = (
            target_density.predict_proba(X_source) @ target_density.weights_
        )

        return {
            DomainNameEnum.source: source_density_at_source_point,
            DomainNameEnum.target: target_density_at_source_point,
        }

    def delete_gmm_root_dir(self) -> None:
        """Delete the temporary folders for GMMs"""
        if self.gmm_root_dir is not None:
            assert "tmp_gmm" in self.gmm_root_dir
            if os.path.exists(self.gmm_root_dir):
                shutil.rmtree(self.gmm_root_dir)

    def preprocess(self, preprocessing_pipeline: List[Preprocessor]) -> None:
        # set the preprocessing pipeline
        self.preprocessing_pipeline = preprocessing_pipeline

        # fit pipeline
        self.preprocessing_pipeline = [
            preprocessor.fit(self.source[DataSplitEnum.train])
            for preprocessor in preprocessing_pipeline
        ]

        # transform source
        for k in self.source:
            for preprocessor in preprocessing_pipeline:
                self.source[k] = preprocessor.transform(self.source[k])

        # add fitting on train of target

        # transform target
        for k in self.target:
            for preprocessor in preprocessing_pipeline:
                self.target[k] = preprocessor.transform(self.target[k])

        return None

    def save_to_disk(self, source_dir: str, target_dir: str) -> None:
        # save the source
        self.source.save_to_disk(source_dir)
        # save the target
        self.target.save_to_disk(target_dir)

    def load_from_disk(self, source_dir: str, target_dir: str) -> TTLDataset:
        self.source = load_from_disk(source_dir)
        self.target = load_from_disk(target_dir)
        return self

    def get_label_binarizer(self) -> OneHotEncoder:
        """Fits the label binarizer and returns it"""
        label_binarizer = OneHotEncoder(sparse_output=False)
        class_label = self.source[DataSplitEnum.train.value].features[DatasetColumnsEnum.labels]
        labels = [[class_label.str2int(s)] for s in class_label.names]
        label_binarizer.fit(labels)
        return label_binarizer

    def get_target_names(self) -> List[str]:
        """Will return the str representation of the classes"""
        class_lbl: ClassLabel = self.source[DataSplitEnum.train].features[DatasetColumnsEnum.labels]
        return [class_lbl.int2str(i) for i in range(class_lbl.num_classes)]
