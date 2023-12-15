import argparse
import os
import json
from pathlib import Path as PathLib
from _utils.stages import Stages
from _utils.paths import Paths
from pydantic import BaseModel
from _utils.enumerations import DatasetNameEnum, TLClassifierEnum, DomainNameEnum
import pandas as pd
import uuid
from typing import List


class Result(BaseModel):

    dataset_name: DatasetNameEnum
    classifier_name: TLClassifierEnum
    params: Stages
    metrics: dict
    xp_name: str
    xp_uuid: str
    path: str
    source_name: str
    target_name: str


def get_missing_xps(res: pd.DataFrame) -> List[dict]:
    """
    Get the dict of missing XPs
    """

    missing = []

    g = res.groupby(["dataset_name", "classifier_name"])[["source_name", "target_name"]].count()

    max_per_dname = g.groupby(["dataset_name"]).max()

    # loop through all combinations of clf and dname
    for dname in res["dataset_name"].unique():
        for cname in res["classifier_name"].unique():
            idx = (dname, cname)

            if idx not in g.index:
                missing.append(
                    {
                        "classifier_name": cname,
                        "dataset_name": dname,
                    }
                )
                continue

            dname = idx[0]

            maxi = max_per_dname.loc[dname, "source_name"]

            if g.loc[idx, "source_name"] < maxi:
                missing.append(
                    {
                        "classifier_name": cname,
                        "dataset_name": dname,
                    }
                )

    return missing


def main(args):
    # loop through all xps and append them results
    xp_dirs = PathLib(args.CLUSTER_ROOT_XPS).glob("*/")

    # the data to save
    res = []

    # loop through all experiments / jobs
    for xp_dir in xp_dirs:
        # get name of xp
        xp_name = str(xp_dir).split("/")[-1]

        # get id of xp
        xp_uuid = str(uuid.uuid1())

        # get the params file path
        params_file = xp_dir.joinpath("src/params.json")

        # if params_file does not exist skip
        if not os.path.exists(params_file):
            continue

        # parse it
        stages_config: Stages = Stages.parse_file(params_file)

        # get the dataset name
        dataset_name = stages_config.preprocess.dataset_name

        # get the classifier name
        classifier_name = stages_config.train.classifier_name

        # source name
        source_name = stages_config.preprocess.dataset_config[DomainNameEnum.source].name

        # target name
        target_name = stages_config.preprocess.dataset_config[DomainNameEnum.target].name

        # get paths
        paths = Paths(stages_config=stages_config)

        # get the results file
        results_file = xp_dir.joinpath("src/").joinpath(paths.get_results_file())

        # if params_file does not exist skip
        if not os.path.exists(results_file):
            continue

        # parse it
        with open(results_file, "r") as f:
            results = json.load(f)

        cur_res = Result(
            dataset_name=dataset_name,
            classifier_name=classifier_name,
            params=stages_config,
            metrics=results,
            xp_uuid=xp_uuid,
            xp_name=xp_name,
            path=args.CLUSTER_ROOT_XPS,
            source_name=source_name,
            target_name=target_name,
        )

        res.append(cur_res.dict())

    # to data frame
    res = pd.DataFrame.from_records(res)

    fname = args.fname if args.fname is not None else paths.aggregated_results_file

    missing = get_missing_xps(res)

    missing_file = fname.replace("results", "missing")

    with open(missing_file, "w") as f:
        json.dump(missing, f, indent=4)

    print("Done missing results in:", missing_file)

    # dump results as append to the results file
    with open(fname, "w") as f:
        res.to_json(f, lines=True, orient="records")

    print("Done jsonl results in:", fname)


if __name__ == "__main__":
    # parse to get the params
    parser = argparse.ArgumentParser()

    # CLUSTER_ROOT_XPS
    parser.add_argument(
        "--CLUSTER_ROOT_XPS",
        required=True,
        default=None,
        dest="CLUSTER_ROOT_XPS",
        help="The path to the cluster directory output.",
    )

    parser.add_argument(
        "--fname",
        required=False,
        dest="fname",
        help="The file name of results - default is results.jsonl",
    )

    # parse the args
    args = parser.parse_args()

    main(args)
