from _datasets.base import TLDataset
from pydantic import BaseModel, ValidationError
from typing import Dict, List, Type
import ray.tune
import pandas as pd
from datasets import disable_caching
import copy
import json
import os
from _utils.stages import RayConfig, TunerConfig
from _utils.enumerations import *
from _classifiers.base import TLClassifier, CVSplit
from _utils.hub import Hub
from _utils.paths import Paths
from _utils.stages import Stages
from itertools import groupby
from collections import defaultdict
from _utils.utils import set_random_seed
import glob
import shutil
from _scoring.base import Scorer
import torch


class GPUStopper(ray.tune.Stopper):
    """
    This stopper will check for each trial if the GPU is available
    Since it might happen that we loose GPU (or nvidia driver)
    during experiments
    """

    def __init__(self, ressources: Dict[RayRessourcesEnum, int]):
        self.ressources = ressources

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        # if gpu is used and cuda is not available
        if (
            self.ressources.get(RayRessourcesEnum.gpu, 0) >= 1
            and torch.cuda.is_available() is False
        ):
            print("gpu/cuda selected yet not available stopping all trials")
            return True

        # otherwise if not gpu or cuda is available do not stop all trials
        return False


class BudgetTimeOutStopper(ray.tune.Stopper):
    """
    This stopper behaves like the official TimeoutStopper
    https://docs.ray.io/en/latest/tune/api_docs/stoppers.html#timeoutstopper-tune-stopper-timeoutstopper
    However it takes into account the workers running in parallel instead of only having a timer on the head.
    """

    budget: float  # the total time budget
    xp_path: str  # param of the tune dir
    tune_eval_paths: str  # the evaluate paths regex
    tune_xp_states: str  # the paths to the xp states json regex

    def __init__(self, xp_path: str, budget: float, tune_eval_paths: str, tune_xp_states: str):
        self.xp_path = xp_path
        self.budget = budget
        self.tune_eval_paths = tune_eval_paths
        self.tune_xp_states = tune_xp_states

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        exp_json_pth = glob.glob(self.tune_xp_states)

        if len(exp_json_pth) == 0:
            # then nothing has started yet
            return False

        # assert len(exp_json_pth) == 1, exp_json_pth
        # exp_json_pth = exp_json_pth[0]
        # read the results
        df = ray.tune.ExperimentAnalysis(experiment_checkpoint_path=self.xp_path).dataframe()

        Tuning.clean_up_tune_dir(self.tune_eval_paths)

        # get total running time
        if df.shape[0] == 0:
            # then no runs finished yet
            return False

        total_time = df["time_this_iter_s"].sum()

        if total_time >= self.budget:
            return True

        return False


class RayCommand(BaseModel):
    config: Dict
    sampling_name: str

    def get_command(self):
        return getattr(ray.tune, self.sampling_name)(**self.config)


class Tuning(BaseModel):
    tuner_config: TunerConfig
    ray_config: RayConfig
    classifier_class: Type[TLClassifier]
    random_seed: int
    paths: Paths
    tl_tuners: List[TLTunerEnum]

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def clean_up_tune_dir(tune_eval_paths: str):
        # clear empty dir since it a known issue in ray
        # https://docs.ray.io/en/latest/_modules/ray/tune/experiment/trial.html
        for foldr in glob.glob(tune_eval_paths):
            if len(os.listdir(foldr)) == 0:
                os.rmdir(foldr)

    @staticmethod
    def get_mode_from_metric(metric_key: MetricKeysEnum):
        """Returns the mode for a certain metric"""
        if metric_key == MetricKeysEnum.reweight_5_loss:
            return "min"
        return "max"

    @staticmethod
    def get_set_hyperparameter_auxiliary(dictionary: Dict) -> Dict:
        """
        :param dictionary: dictionary of variable depth that contains details for the ray sampling
        Recursive auxilliary function associated with get_set_hyperparameter.
        """
        for key in dictionary:
            try:
                # create a raycommand using the name and the config
                ray_command = RayCommand(**dictionary)

                # get the command associate back to the dictionary
                dictionary = ray_command.get_command()

                # stop the loop as the associated command as already been set
                return dictionary

            except ValidationError:
                # continue recursively on all the dictionary.
                dictionary[key] = Tuning.get_set_hyperparameter_auxiliary(dictionary[key])

        return dictionary

    @staticmethod
    def get_analysis(tune_dir: str) -> pd.DataFrame:
        """
        :param tune_dir: Directory of the ray experiment.
        """
        return ray.tune.ExperimentAnalysis(tune_dir).dataframe()

    @staticmethod
    def get_best_hyperparam(stages_config: Stages) -> Dict:
        """
        :param stages_config: The stages config.
        """
        # instanciate the paths
        paths: Paths = Paths(stages_config=stages_config)

        # find the metric and the mode
        search_method_name = stages_config.train.tune_config.search_method_name

        # load the best cfg dict
        with open(paths.get_ray_best_cfgs_pth()) as f:
            cfgs = json.load(f)

        # return the best cfg
        return cfgs.get(search_method_name)

    def get_list_tl_dataset_splits(self, tl_dataset) -> List[TLDataset]:
        """
        This function will generate the splitted tl datasets for hparam tune
        """
        # the res
        res = []

        # get the splits
        cv_split: CVSplit = TLClassifier.get_splits_for_cv(
            tl_dataset=tl_dataset, model_selection=self.tuner_config.model_selection
        )

        # loop through the splits
        while True:
            # try next on the splits
            try:
                # get the new dataset
                new_tl_dataset: TLDataset = TLClassifier.get_new_tl_dataset_based_on_splits(
                    tl_dataset=tl_dataset,
                    split_source=next(cv_split.source),
                    split_target=next(cv_split.target),
                )

                # append it
                res.append(new_tl_dataset)

            except StopIteration:
                # read all splits finish loop
                break

        # return the list of tl datasets
        return res

    def fit(self, tl_dataset: TLDataset) -> pd.DataFrame:
        """
        Select and launch all the hyperparameters in parralele.
        """
        # fix those value outside of the class so that ray can access them
        # classifier_class = self.config["classifier_class"]
        # metric_names = self.config["metric_names"]

        # get the list of tl_dataset
        new_tl_dataset_s: List[TLDataset] = self.get_list_tl_dataset_splits(tl_dataset=tl_dataset)

        assert (
            len(new_tl_dataset_s) == 1
        ), "fitting gmm here won't work if we have several dataset splits"

        # fit the gaussian model if need

        # loop through all dataset splits
        for i in range(len(new_tl_dataset_s)):
            Scorer(tl_dataset=new_tl_dataset_s[i]).fit_gmm_if_needed(tl_dataset=new_tl_dataset_s[i])

        def evaluate(
            hyperparam_tuning: Dict,
            classifier_class: Type[TLClassifier] = self.classifier_class,
            hyperparam_fixed: Dict = self.tuner_config.hyperparam_fixed,
            random_seed: int = self.random_seed,
        ) -> Dict[TLTunerEnum, Dict[str, Dict]]:
            """
            :param config_hyperparam: one set of hyperparameter that is selected by ray
            Function launched by the ray head to each worker. It reports the error
            """
            # disable caching in huggingface
            disable_caching()

            set_random_seed(random_seed)

            # merge the two dicts
            classifier_hparams = TLClassifier.merge_dicts(
                new=copy.deepcopy(hyperparam_tuning),
                res=copy.deepcopy(hyperparam_fixed),
                allow_overwrite=False,
            )

            # dict that will contain the metrics
            reported_metrics = defaultdict(dict)

            # loop through the splits
            for split_idx in range(len(new_tl_dataset_s)):

                print(f"Fitting split: {split_idx}")

                # get the new dataset
                new_tl_dataset: TLDataset = new_tl_dataset_s[split_idx]

                # initialize the classifier with hyperparams
                classifier: TLClassifier = classifier_class(
                    classifier_hparams,
                    train_dir="./tmp_trainer",  # no need to save load model when hparams tuning
                    pred_dir="./tmp_trainer",  # no need to save load model when hparams tuning
                    tl_dataset=new_tl_dataset,
                )

                # fit the classifier
                classifier = classifier.fit(new_tl_dataset)

                # results
                reported_metric = classifier.evaluate(new_tl_dataset)

                for cur_key in DomainNameEnum:

                    # update the reported metrics by inserting the cv split
                    reported_metrics[cur_key.value].update(
                        {f"split-{split_idx}": reported_metric[cur_key.value]}
                    )

            # compute average over splits for each TLTuner
            reported_metrics = Tuning.compute_average_metrics_per_tl_tuner(reported_metrics)

            # report the metrics to the head of the ray cluster
            return reported_metrics

        # find and attach tot the, potentialy distant, ray cluster address="auto"
        ray.init(address=self.tuner_config.address)

        # get the dictionary of all ray parameter
        ray_config_dict = self.ray_config.dict()

        # remove the time_budget parameter (in hours)
        ray_config_dict.pop("time_budget")

        budget_stopper = BudgetTimeOutStopper(
            xp_path=self.paths.get_tune_xp_path(),
            budget=ray_config_dict.pop("time_budget_s"),
            tune_eval_paths=self.paths.get_tune_eval_paths(),
            tune_xp_states=self.paths.get_tune_state_xp(),
        )

        gpu_stopper = GPUStopper(ressources=self.ray_config.resources_per_trial)

        stopper = ray.tune.stopper.CombinedStopper(gpu_stopper, budget_stopper)

        # Transform a dictionary that specify the ray sampling method and range
        # to a RayCommand version of the dictionary.
        # copy the dictionary to keep a clean version in self.config
        dict_hyperparam = copy.deepcopy(self.tuner_config.hyperparam_tuning)
        # transform the dict
        dict_hyperparam = self.get_set_hyperparameter_auxiliary(dict_hyperparam)

        # main ray function that search over the set of hyperparameters
        # and launch the evaluation function
        # in parralele for every set of hyperparameter
        analysis = ray.tune.run(evaluate, stop=stopper, config=dict_hyperparam, **ray_config_dict)

        # remove the tmp gmm dir
        for j in range(len(new_tl_dataset_s)):
            new_tl_dataset_s[j].delete_gmm_root_dir()

        Tuning.clean_up_tune_dir(tune_eval_paths=self.paths.get_tune_eval_paths())

        # copy exp states that will be tracked by dvc instead of tracking everything
        os.makedirs(self.paths.get_tune_final(), exist_ok=True)
        for state_exp in glob.glob(self.paths.get_tune_state_xp()):
            src = state_exp
            dest = self.paths.get_path_state_final(state_exp.split("/")[-1])
            shutil.copyfile(state_exp, dest)

        # return a dataframe that sumarize the experiment
        ray_df = analysis.dataframe()
        self._check_if_ray_trials_all_failed(ray_df)

        # dump ray df in final path
        ray_df.to_csv(self.paths.get_ray_df_pth())

        # dump best cfgs
        self._dump_best_cfgs(ray_xp=analysis)

        return ray_df

    def _dump_best_cfgs(self, ray_xp: ray.tune.ExperimentAnalysis) -> None:
        """
        Will dump the best confgis into a dict with key tuner name and value the best cfg dict
        """
        # compute ray best configs
        best_cfgs = {}
        # loop through tl tuner
        for tl_tuner_name in self.tl_tuners:
            # get the tuner obj
            tuner = Hub.get_tl_tuner_class(tl_tuner_name)()

            # get the mode associated with the metric used
            mode = Tuning.get_mode_from_metric(tuner.get_metric_key_if_none(None))

            # get the best config dict
            cfg: dict = ray_xp.get_best_config(tuner.get_metric_key(None), mode)

            # fill the best cfg
            best_cfgs[tl_tuner_name.value] = cfg

        # dump to json
        with open(self.paths.get_ray_best_cfgs_pth(), "w") as f:
            json.dump(best_cfgs, f)

    def _check_if_ray_trials_all_failed(self, ray_df: pd.DataFrame) -> None:
        """Raise error if all ray trials status is error"""
        if len(ray_df) == 0:
            raise Exception("All ray trials raised an error.")

    @staticmethod
    def average_dict(d: Dict) -> Dict:
        """
        Average a nested dict of dicts.
        Ref:
        https://stackoverflow.com/questions/57311453/calculate-average-values-in-a-nested-dict-of-dicts
        """
        _data = sorted([i for b in d for i in b.items()], key=lambda x: x[0])
        _d = [(a, [j for _, j in b]) for a, b in groupby(_data, key=lambda x: x[0])]
        return {
            a: Tuning.average_dict(b)
            if isinstance(b[0], dict)
            else round(sum(b) / float(len(b)), 2)
            for a, b in _d
        }

    def compute_average_metrics_per_tl_tuner(reported_metrics: Dict) -> Dict:
        res = copy.deepcopy(reported_metrics)
        for cur_key in res:
            # average for all metrics
            res[cur_key].update(
                {AverageEnum.average.value: Tuning.average_dict(list(res[cur_key].values()))}
            )
        return res
