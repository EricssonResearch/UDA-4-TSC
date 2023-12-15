from _utils.utils import set_random_seed
from _classifiers.base import TLClassifier
from _utils.stages import Stages
from _utils.paths import Paths
from _utils.enumerations import *
from transformers import EvalPrediction
from datasets import disable_caching
import argparse
import pickle
from typing import Dict
from _utils.hub import Hub
from _tuners.base import TLTuner


def dump_predictions(
    stages_config: Stages, predictions: Dict[DomainNameEnum, Dict[str, EvalPrediction]]
) -> None:
    # set the output file
    out_file: str = Paths(stages_config=stages_config).get_pred_pkl()

    # dump pkl file
    with open(out_file, "wb") as f:
        pickle.dump(predictions, f)


def main(args):
    # disable caching in huggingface
    disable_caching()

    # get the config file path
    json_file_path = args.configFile

    # read the json params file
    stages_config: Stages = Stages.parse_file(json_file_path)

    # set random seed
    set_random_seed(stages_config.predict.random_seed)

    # load TLDataset from disk that is already preprocessed
    tl_dataset = TLClassifier.load_transfer_learning_dataset_from_disk(stages_config)

    # perform prediction for every search method in tune stage
    for search_method_name in stages_config.tune.search_method_names:
        # fill how to choose the best epoch depending on search method
        tltuner: TLTuner = Hub.get_tl_tuner_class(search_method_name)()
        tltuner.fill_load_model(stages_config.predict.load_model)

        # update stages_config with the current search method name
        stages_config.train.tune_config.set_search_method_name(search_method_name)

        # update the classifier parameter using the best parameters if necessary
        new_stages_config = TLClassifier.get_best_param(stages_config)

        # get classifier based on config
        classifier: TLClassifier = TLClassifier.get_classifier(new_stages_config, tl_dataset)

        # load the trained model for the classifier
        classifier.load_model(**new_stages_config.predict.load_model.dict())

        # perform prediction
        predictions = classifier.predict(tl_dataset)

        # dump prediction
        dump_predictions(new_stages_config, predictions)

        # put back to None the checkpiont configs
        tltuner.unfill_load_model(stages_config.predict.load_model)

    print("Done predictions")


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

    main(args)
