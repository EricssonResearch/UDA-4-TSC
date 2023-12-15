from _datasets.base import TLDataset
from typing import TypeVar, Dict, Union, Tuple, Generator
import abc
from transformers import EvalPrediction
from _utils.enumerations import *
from _utils.stages import (
    Stages,
    ModelSelection,
    NoShuffleSplitConfig,
)
from _utils.paths import Paths
import os
import copy
from _scoring.base import Scorer
import numpy as np
from datasets import DatasetDict
from sklearn.model_selection import BaseShuffleSplit
from sklearn.utils.validation import _num_samples


class NoShuffleSplit(BaseShuffleSplit):
    """Split but do not shuffle, meaning first part will be train and the rest test"""

    def __init__(
        self,
        n_splits,
        test_size,
    ):
        assert n_splits == 1, "Only a single split is allowed"
        assert 0.0 < test_size < 1.0, "test_size should be a float between 0 and 1"
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
        )
        self._default_test_size = 0.1

    def _iter_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        n_test = int(n_samples * self.test_size)
        n_train = n_samples - n_test

        for i in range(self.n_splits):
            idx = np.arange(n_samples)
            ind_train = idx[:n_train]
            ind_test = idx[n_train : (n_test + n_train)]
            yield ind_train, ind_test


class CVSplit:

    source: Generator[Tuple[np.array, np.array], None, None]
    target: Generator[Tuple[np.array, np.array], None, None]

    def __init__(
        self,
        source: Generator[Tuple[np.array, np.array], None, None],
        target: Generator[Tuple[np.array, np.array], None, None],
    ):
        self.source = source
        self.target = target


TTLClassifier = TypeVar("TTLClassifier", bound="TLClassifier")


class TLClassifier(abc.ABC):

    config: Dict
    train_dir: str
    pred_dir: str

    def __init__(
        self,
        config: Dict,
        train_dir: str,
        pred_dir: str,
        tl_dataset: TLDataset,
    ):
        self.config = self.inject_config_based_on_dataset(config=config, tl_dataset=tl_dataset)
        self.train_dir = train_dir
        self.pred_dir = pred_dir

    def inject_config_based_on_dataset(self, config: Dict, tl_dataset: TLDataset) -> Dict:
        """For example if needed will inject num_labels"""
        return config

    @abc.abstractmethod
    def fit(self, tl_dataset: TLDataset) -> TTLClassifier:
        pass

    def evaluate(
        self,
        tl_dataset: TLDataset,
    ) -> Dict[DomainNameEnum, Dict[DataSplitEnum, Dict[MetricKeysEnum, float]]]:
        """
        This functions supposes that you have already loaded your model
        """
        # get predictions
        predictions = self.predict(tl_dataset)
        # get scorer
        scorer = Scorer(tl_dataset=tl_dataset)
        # return computed metrics
        return scorer.compute_metrics(predictions)

    @abc.abstractmethod
    def save_model(self, **args) -> None:
        pass

    @abc.abstractmethod
    def load_model(self, **args) -> None:
        pass

    @abc.abstractmethod
    def predict(self, tl_dataset: TLDataset) -> Dict[DomainNameEnum, Dict[str, EvalPrediction]]:
        """
        For each dataset (source, target) in the TLDataset, predict for both splits (train, test)
        """
        pass

    @staticmethod
    def load_transfer_learning_dataset_from_disk(stages_config: Stages) -> TLDataset:
        paths = Paths(stages_config=stages_config)
        return TLDataset(
            dataset_name=stages_config.preprocess.dataset_name, source=None, target=None
        ).load_from_disk(
            source_dir=paths.get_dataset_disk_dir_source(),
            target_dir=paths.get_dataset_disk_dir_target(),
        )

    @staticmethod
    def get_classifier(stages_config: Stages, tl_dataset: TLDataset) -> TTLClassifier:
        from _utils.hub import Hub

        # get classifier name from config
        classifier_name = stages_config.train.classifier_name

        # get classifier's config from the whole config
        classifier_config = stages_config.train.config

        # create the paths
        paths = Paths(stages_config=stages_config)

        # set the training out dir
        train_dir = paths.get_train_dir()

        # create the dir if it does not exist
        os.makedirs(train_dir, exist_ok=True)

        # set the prediction out dir
        pred_dir = paths.get_pred_dir()

        # create the dir if it does not exist
        os.makedirs(pred_dir, exist_ok=True)

        # get the classifier's class to instantiate
        classifier_class = Hub.get_tl_classifier_class(classifier_name=classifier_name)

        # return the instance
        return classifier_class(
            config=classifier_config, train_dir=train_dir, pred_dir=pred_dir, tl_dataset=tl_dataset
        )

    @staticmethod
    def merge_dicts(new: Dict, res: Dict, allow_overwrite: bool) -> Dict:
        """
        This function merges two dicts res and new. Basically
        it recursively does res.update(new) then returns res
        param `allow_overwrite`: if False will raise error if
        a key exists in both dicts
        """
        for key in new:
            if isinstance(new[key], dict):
                res[key] = TLClassifier.merge_dicts(
                    new=new[key],
                    res=res.get(key, {}),
                    allow_overwrite=allow_overwrite,
                )
            else:
                if (allow_overwrite is False) and (key in res):
                    raise Exception("Train and Tune configs should not have parameters in common")
                res[key] = new[key]
        return res

    @staticmethod
    def get_best_param(stages_config_original: Stages) -> Stages:
        from _tuners.Tuning import Tuning

        # do not fetch best param if not needed
        if (
            stages_config_original.train.tune_config.train_tune_option
            == TrainTuneOptionsEnum.train_configs_only
        ):
            return stages_config_original

        # copy instance of stages configs
        stages_config = copy.deepcopy(stages_config_original)

        # find the best param from tuning
        tune_config = Tuning.get_best_hyperparam(stages_config=stages_config)

        # get default hyparams in tune stage
        fixed_hparams = stages_config.tune.tuner_config.hyperparam_fixed

        # merge it with the default dict of hparams
        tune_config = TLClassifier.merge_dicts(
            new=copy.deepcopy(tune_config),
            res=copy.deepcopy(fixed_hparams),
            allow_overwrite=False,
        )

        # get the train config
        train_config = copy.deepcopy(stages_config.train.config)

        # get the option for merging
        train_tune_option = stages_config.train.tune_config.train_tune_option

        # double check how merging train configs and tune configs
        if train_tune_option == TrainTuneOptionsEnum.tune_configs_only:
            print(stages_config.train.config)
            # only using tune configs
            assert (
                len(stages_config.train.config) == 0
            ), "with option tune_configs_only train.config should be empty"
            # replace the train dictionary by the best one found during tuning
            stages_config.train.config = tune_config
        elif train_tune_option == TrainTuneOptionsEnum.train_union_tune:
            # tune will be merged with train but no intersection allowed
            stages_config.train.config = TLClassifier.merge_dicts(
                new=train_config, res=tune_config, allow_overwrite=False
            )
        elif train_tune_option == TrainTuneOptionsEnum.train_overrides_tune:
            # train will override tune config
            stages_config.train.config = TLClassifier.merge_dicts(
                new=train_config, res=tune_config, allow_overwrite=True
            )
        elif train_tune_option == TrainTuneOptionsEnum.tune_overrides_train:
            # tune will override train config
            stages_config.train.config = TLClassifier.merge_dicts(
                new=tune_config, res=stages_config.train.config, allow_overwrite=True
            )
        else:
            raise Exception(f"Not expected option {train_tune_option}")

        return stages_config

    @staticmethod
    def get_splits_for_cv(tl_dataset: TLDataset, model_selection: ModelSelection) -> CVSplit:
        """
        Given a dataset generate the folds for cross validation
        """
        from _utils.hub import Hub

        folds = Hub.get_model_selection(model_selection.name)(**model_selection.config)

        # get the splits for source
        splits_source = folds.split(
            np.zeros(tl_dataset.source[DataSplitEnum.train.value].num_rows),
            tl_dataset.source[DataSplitEnum.train.value][DatasetColumnsEnum.labels],
        )

        # get the splits for target
        splits_target = folds.split(
            np.zeros(tl_dataset.target[DataSplitEnum.train.value].num_rows),
            tl_dataset.target[DataSplitEnum.train.value][DatasetColumnsEnum.labels],
        )

        # return tuple for each source and target
        return CVSplit(source=splits_source, target=splits_target)

    @staticmethod
    def get_new_tl_dataset_based_on_splits(
        tl_dataset: TLDataset,
        split_source: Tuple[np.array, np.array],
        split_target: Tuple[np.array, np.array],
    ) -> TLDataset:
        # prepare the source
        new_source: DatasetDict = DatasetDict(
            {
                DataSplitEnum.train.value: tl_dataset.source[DataSplitEnum.train.value].select(
                    split_source[0]  # 0 for train
                ),
                DataSplitEnum.test.value: tl_dataset.source[DataSplitEnum.train.value].select(
                    split_source[1]  # 1 for val
                ),
            }
        )

        # prepare the target
        new_target: DatasetDict = DatasetDict(
            {
                DataSplitEnum.train.value: tl_dataset.target[DataSplitEnum.train.value].select(
                    split_target[0]  # 0 for train
                ),
                DataSplitEnum.test.value: tl_dataset.target[DataSplitEnum.train.value].select(
                    split_target[1]  # 1 for val
                ),
            }
        )

        # create the new tl dataset
        new_tl_dataset: TLDataset = TLDataset(
            dataset_name=tl_dataset.dataset_name,
            preprocessing_pipeline=tl_dataset.preprocessing_pipeline,
            source=new_source,
            target=new_target,
        )

        return new_tl_dataset

    @staticmethod
    def get_tl_dataset_with_val_instead_of_test(
        tl_dataset: TLDataset, model_selection_config: NoShuffleSplitConfig
    ) -> TLDataset:
        # split if needed
        if model_selection_config.n_splits == 1:
            print("Splitting train into train/val")
            # create the model selection based on the shufflesplit only
            model_selection = ModelSelection(
                name=ModelSelectionEnum.NoShuffleSplit, config=model_selection_config.dict()
            )
            # perform split
            cv_split: CVSplit = TLClassifier.get_splits_for_cv(
                tl_dataset=tl_dataset, model_selection=model_selection
            )

            # get the new dataset
            return TLClassifier.get_new_tl_dataset_based_on_splits(
                tl_dataset=tl_dataset,
                split_source=next(cv_split.source),
                split_target=next(cv_split.target),
            )
        elif model_selection_config.n_splits == 0:
            print("no split is done -> use train as train and val")
            # source splits means only source.train indexes
            split_source_train = np.arange(
                start=0, stop=tl_dataset.source[DataSplitEnum.train].num_rows
            )
            # target splits means only target.train indexes
            split_target_train = np.arange(
                start=0, stop=tl_dataset.target[DataSplitEnum.train].num_rows
            )
            return TLClassifier.get_new_tl_dataset_based_on_splits(
                tl_dataset=tl_dataset,
                split_source=(split_source_train, split_source_train),
                split_target=(split_target_train, split_target_train),
            )
        else:
            # only n_split 0 or 1 are supported
            raise Exception("config.model_selection_config.n_splits either 0 or 1")
