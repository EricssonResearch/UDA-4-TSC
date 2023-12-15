"""
This script will generate the upper bound in configs/custom_conf/*dataset_name* 
"""
import copy
import json
from typing import Dict, List
from _utils.dvc import SampleConfig, DatasetConfig
from _utils.enumerations import *


if __name__ == "__main__":
    dataset_names = [
        DatasetNameEnum.har,
        DatasetNameEnum.hhar,
        DatasetNameEnum.mfd,
        DatasetNameEnum.sleepStage,
        DatasetNameEnum.sportsActivities,
        DatasetNameEnum.ultrasoundMuscleContraction,
        DatasetNameEnum.wisdm,
    ]

    root_dir = "_configs/custom_conf/dataset_name={}/sample.json"

    for dname in dataset_names:
        # read sample json
        fname = root_dir.format(dname.value)
        sample: SampleConfig = SampleConfig.parse_file(fname)

        extra_dataset_configs: List[Dict[DomainNameEnum, DatasetConfig]] = []

        for dataset_conf in sample.dataset_configs:
            upper_dataset_conf: Dict[DomainNameEnum, DatasetConfig] = {}

            # swap source and target
            upper_dataset_conf[DomainNameEnum.source] = copy.deepcopy(
                dataset_conf[DomainNameEnum.target]
            )
            upper_dataset_conf[DomainNameEnum.target] = copy.deepcopy(
                dataset_conf[DomainNameEnum.source]
            )

            # add to extra list
            extra_dataset_configs.append(upper_dataset_conf)

        # extend the original list with the extra list
        sample.dataset_configs.extend(extra_dataset_configs)

        # dump the sample json
        with open(fname, "w") as f:
            json.dump(sample.dict(), f, indent=4)
