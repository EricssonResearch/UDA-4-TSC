from _utils.utils import set_random_seed
from datasets import disable_caching, load_dataset, DatasetDict
import argparse
from _utils.stages import Stages, PreprocessorConfig
from _utils.enumerations import DatasetNameEnum, DomainNameEnum, DataSplitEnum, DatasetColumnsEnum
from _utils.paths import Paths
from _utils.hub import Hub
from typing import Dict, List
from _datasets.base import TLDataset
from _preprocessing.Preprocessor import Preprocessor


def build_preprocessing_pipeline(stages_config: Stages) -> List[Preprocessor]:
    preprocessor_configs: List[PreprocessorConfig] = stages_config.preprocess.preprocessing_pipeline
    preprocessing_pipeline = []
    for preprocessor_config in preprocessor_configs:
        preprocessor_class = Hub.get_preprocessor_class(preprocessor_config.preprocessor)
        config = preprocessor_config.config
        preprocessing_pipeline.append(preprocessor_class(**config))

    return preprocessing_pipeline


def save_tl_dataset_to_disk(tl_dataset: TLDataset, stages_config: Stages) -> None:
    paths = Paths(stages_config=stages_config)
    tl_dataset.save_to_disk(
        source_dir=paths.get_dataset_disk_dir_source(),
        target_dir=paths.get_dataset_disk_dir_target(),
    )


def load_transfer_learning_dataset(stages_config: Stages) -> TLDataset:
    dataset_name: DatasetNameEnum = stages_config.preprocess.dataset_name
    dataset_config: Dict[DomainNameEnum, Dict] = stages_config.preprocess.dataset_config

    paths = Paths(stages_config=stages_config)
    dataset_path = paths.get_dataset_path()

    source: DatasetDict = load_dataset(dataset_path, **dataset_config[DomainNameEnum.source].dict())
    target: DatasetDict = load_dataset(dataset_path, **dataset_config[DomainNameEnum.target].dict())

    # assert that splits train and test are defined in source and target
    assert (
        (DataSplitEnum.train in source)
        and (DataSplitEnum.test in source)
        and (DataSplitEnum.train in target)
        and (DataSplitEnum.test in target)
    ), "Make sure you have train and test splits defined in your source and target datasets"

    # assert that the columns are correctly defined in the datasets for source and target
    assert (
        (DatasetColumnsEnum.labels in source[DataSplitEnum.train].column_names)
        and (DatasetColumnsEnum.mts in source[DataSplitEnum.train].column_names)
        and (DatasetColumnsEnum.labels in source[DataSplitEnum.test].column_names)
        and (DatasetColumnsEnum.mts in source[DataSplitEnum.test].column_names)
        and (DatasetColumnsEnum.labels in target[DataSplitEnum.train].column_names)
        and (DatasetColumnsEnum.mts in target[DataSplitEnum.train].column_names)
        and (DatasetColumnsEnum.labels in target[DataSplitEnum.test].column_names)
        and (DatasetColumnsEnum.mts in target[DataSplitEnum.test].column_names)
    ), "source and target should have both the columns 'mts' and 'labels'"

    return TLDataset(dataset_name=dataset_name, source=source, target=target)


def main(args):
    # disable caching in huggingface
    disable_caching()

    # get the config file path
    json_file_path = args.configFile

    # read the json params file
    stages_config: Stages = Stages.parse_file(json_file_path)

    # set random seed
    set_random_seed(stages_config.preprocess.random_seed)

    # load the transfer learning dataset
    tl_dataset = load_transfer_learning_dataset(stages_config)

    # build the preprocessing pipeline
    preprocessing_pipeline = build_preprocessing_pipeline(stages_config)

    # apply the pipeline (includes fitting on source[train]  and transforming the rest)
    tl_dataset.preprocess(preprocessing_pipeline)

    # save the preprocessed dataset
    save_tl_dataset_to_disk(tl_dataset=tl_dataset, stages_config=stages_config)


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
