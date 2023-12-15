from _utils.utils import set_random_seed
from _utils.stages import EmptyDict, Stages
from _utils.hub import Hub
from _utils.paths import Paths
from datasets import disable_caching
from _tuners.Tuning import Tuning
from _classifiers.base import TLClassifier
import argparse
import os


def dump_file_for_emtpy_tune_stage(stages_config: Stages) -> None:
    paths = Paths(stages_config=stages_config)
    os.makedirs(paths.get_tune_final(), exist_ok=True)
    out_file = paths.get_empty_tune_file()
    with open(out_file, "w") as f:
        f.write("empty tune stage")


def get_tuning(stages_config: Stages) -> Tuning:

    # get tune's config from the whole config
    tuner_config = stages_config.tune.tuner_config

    # create the paths
    paths = Paths(stages_config=stages_config)

    # verify that the train and tune classifier are the same to avoid saving path issue
    assert (
        stages_config.train.classifier_name.value == stages_config.tune.classifier_name.value
    ), "The classifier name of tune and train should be the same."

    # set the tune out dir
    tune_dir = paths.get_tune_dir()

    # create the dir if it does not exist
    os.makedirs(tune_dir, exist_ok=True)

    # get classifier name
    classifier_name = stages_config.tune.classifier_name

    # get classifier
    classifier_class = Hub.get_tl_classifier_class(classifier_name=classifier_name)

    # get the ray_config
    ray_config = stages_config.tune.ray_config

    # set the tune_dir
    ray_config.local_dir = tune_dir

    # get random seed for ray workers
    random_seed = stages_config.tune.random_seed

    # get the classifier's class to instantiate
    return Tuning(
        tuner_config=tuner_config,
        ray_config=ray_config,
        classifier_class=classifier_class,
        random_seed=random_seed,
        paths=paths,
        tl_tuners=stages_config.tune.search_method_names,
    )


def main(args):
    # disable caching in huggingface
    disable_caching()

    # get the config file path
    json_file_path = args.configFile

    # read the json params file
    stages_config: Stages = Stages.parse_file(json_file_path)

    # set random seed
    set_random_seed(stages_config.tune.random_seed)

    # stop the pipeline if no tuning is required
    if isinstance(stages_config.tune, EmptyDict):
        print("No tuning of hyperparameter done since tune stage is an EmptyDict")
        dump_file_for_emtpy_tune_stage(stages_config=stages_config)
        return

    # load TLDataset from disk that is already preprocessed
    tl_dataset = TLClassifier.load_transfer_learning_dataset_from_disk(stages_config)

    # init the prepocess for the tuning pipeline
    tuning = get_tuning(stages_config)

    # fit all the tuners
    analysis = tuning.fit(tl_dataset)

    print("Done tuning of hyperparameter.")
    print("Size of ray analysis dataframe:", analysis.shape)


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
