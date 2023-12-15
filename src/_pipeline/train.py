from _utils.utils import set_random_seed
from _utils.enumerations import *
from _utils.stages import Stages
from _classifiers.base import TLClassifier
from datasets import disable_caching
from _scoring.base import Scorer
import argparse


def main(args):
    # disable caching in huggingface
    disable_caching()

    # get the config file path
    json_file_path = args.configFile

    # read the json params file
    stages_config: Stages = Stages.parse_file(json_file_path)

    # set random seed
    set_random_seed(stages_config.train.random_seed)

    # load TLDataset from disk that is already preprocessed
    tl_dataset = TLClassifier.load_transfer_learning_dataset_from_disk(stages_config)

    # make sure dataset test set is correctly overriden with val
    tl_dataset = TLClassifier.get_tl_dataset_with_val_instead_of_test(
        tl_dataset, stages_config.train.no_shuffle_split_config
    )

    # make sure the gmm is fit
    Scorer(tl_dataset=tl_dataset).fit_gmm_if_needed(tl_dataset=tl_dataset)

    # perform training for every search method in tune stage
    for search_method_name in stages_config.tune.search_method_names:
        # set random seed for each method
        set_random_seed(stages_config.train.random_seed)

        # update stages_config with the current search method name
        stages_config.train.tune_config.set_search_method_name(search_method_name)

        # update the classifier parameter using the best parameters if necessary
        new_stages_config = TLClassifier.get_best_param(stages_config)

        # get classifier based on config
        classifier: TLClassifier = TLClassifier.get_classifier(new_stages_config, tl_dataset)

        # fit the classifier
        classifier = classifier.fit(tl_dataset)

        # save the model
        classifier.save_model()

    # delete the gmm
    tl_dataset.delete_gmm_root_dir()

    print("Done training.")


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
