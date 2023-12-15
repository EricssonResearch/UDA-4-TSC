from _utils.utils import set_random_seed
from datasets import disable_caching
import argparse
from _scoring.base import Scorer
from _utils.enumerations import *
from _utils.stages import Stages
from _utils.paths import Paths
from typing import Union, Dict
import pickle
import os
import json
from transformers import EvalPrediction
from _classifiers.base import TLClassifier


def dump_metrics(
    metrics: Dict[DomainNameEnum, Dict[str, Dict[str, object]]], stages_config: Stages
) -> None:
    """For each (domain_name, split) create a json."""
    paths = Paths(stages_config=stages_config)

    # get the results json file
    out_dir = paths.get_results_dir()
    out_file = paths.get_results_file()

    # create the directory if not exists
    os.makedirs(out_dir, exist_ok=True)

    # dump the json
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)


def load_predictions(
    stages_config: Stages,
) -> Dict[DomainNameEnum, Dict[str, EvalPrediction]]:
    # get the file path
    file_path: str = Paths(stages_config=stages_config).get_pred_pkl()

    # load pkl file
    with open(file_path, "rb") as f:
        return pickle.load(f)


def main(args):
    # disable caching in huggingface
    disable_caching()

    # get the config file path
    json_file_path = args.configFile

    # read the json params file
    stages_config: Stages = Stages.parse_file(json_file_path)

    # set random seed
    set_random_seed(stages_config.score.random_seed)

    # load TLDataset from disk that is already preprocessed
    tl_dataset = TLClassifier.load_transfer_learning_dataset_from_disk(stages_config)

    # make sure the gmm is fit
    Scorer(tl_dataset=tl_dataset).fit_gmm_if_needed(tl_dataset=tl_dataset)

    # get the scorer and create it
    scorer = Scorer(tl_dataset=tl_dataset)

    all_metrics: Dict[Union[SearchMethodNoneEnum, TLTunerEnum], Dict] = {}

    # perform training for every search method in tune stage
    for search_method_name in stages_config.tune.search_method_names:
        # update stages_config with the current search method name
        stages_config.train.tune_config.set_search_method_name(search_method_name)

        # read the predictions we want to score
        predictions = load_predictions(stages_config)

        # compute the metrics for this search method
        metrics = scorer.compute_metrics(predictions)

        # add it to the dict
        all_metrics[stages_config.train.tune_config.search_method_name] = metrics

    # delete the gmm
    tl_dataset.delete_gmm_root_dir()

    # dump json metrics
    dump_metrics(metrics=all_metrics, stages_config=stages_config)

    print("Done scoring")


if __name__ == "__main__":

    # parser to get the json config file path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configFile",
        default="params.json",
        dest="configFile",
        help="The path to the params.json file that contains the parameters for the pipeline",
    )

    # parse the args
    args = parser.parse_args()

    main(args=args)
