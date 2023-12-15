# Contributing

This is a guide to follow for contributing to UDA-4-TSC repository.

## Pre-commit

We are using [pre-commit](https://pre-commit.com/) package.

After you install the requirements in your conda environment you will be able to install the `git hook` necessary using the following command `pre-commit install`.

From now on every commit you perform will be run through `black` which is defined in `.pre-commit-config.yaml`

If you would like to test the `black` format yourself you can use the command `black --check --line-length 100 .`

## Add a new dataset

When adding a new dataset you will need:
- Create a new dataset file that inherits from hugginface datasets in `_datasets` following the huggingface [tutorial](https://huggingface.co/docs/datasets/v1.11.0/add_dataset.html) on adding a custom dataset.
- Update the base enumeration `DatasetNameEnum` in `src/_utils/enumerations.py`.
- Add a new `_configs/conf/stages/preprocess/[dataset_name].yaml` that defines the preprocessing stage of the dataset you are adding.
- Add the set of different `dataset_configs` to specify what target and source we need in sample configuration in `_configs/custom_conf/dataset_name=[dataset_name]/sample.json`.

### Config for preprocess stage

The config for a given dataset in `_configs/conf/stages/preprocess/[dataset_name].yaml` needs to follow this schema:
- `dataset_name`: the name of the dataset.
- `preprocessing_pipeline`: a list of `PreprocessorConfig` that contains:
    - `preprocessor`: the name of the preprocessor `PreprocessorEnum`.
    - `config`: a dict that contains the custom configuration expected by the preprocessor defined in `src/_preprocessing`.
- `dataset_config`: a dict that contains two other dicts `source` and `target` where each one of these two will need a dict with:
    - `name`: the name of the `source/target`.
    - any other custom keys and values that are needed to configure the `source/target`.

## Add a new classifier

When adding a new classifier you will need:
- Create a new classifier that either inherits from `SKClassifier` in `src/_classifiers/sk/base.py` or inherits from `HuggingFaceClassifier` in `src/_classifiers/hf/base.py`.
    - Note that the config of a `HuggingFaceClassifier` will need to follow the `HFConfig` present in `src/_classifiers/hf/base.py`.
    - The latter `HFConfig` will need a `backbone` attribute / dict configuration to create the backbone defined in `src/_backbone/nn`. 
- Update eh base enumeration `TLClassifierEnum` by inserting your new classifier's name in `src/_utils/enumerations.py`.
- Update the function `get_tl_classifier_class` that fetches the classifier based on its name in `src/_utils/hub.py`.
- For each classifier you will need to define its default configuration for tuning in `_configs/conf/stages/tune/[classifier_name].yaml` and for training in `_configs/conf/stages/train/[classifier_name].yaml`.
- If needed to override the default values for this classifier for some dataset, you need define them in `_configs/conf/stages/tune/tuner_config/[classifier_name]/[dataset_name].yaml` for tune stage and in `_configs/conf/stages/train/config/[classifier_name]/[dataset_name].yaml` for train stage.


### Config for tune stage 

**Skip tune stage**

If your classifier does not need to define its hyperparameters tuning stage, then you are allowed to skip it by putting in `_configs/conf/stages/tune/[classifier_name].yaml` the following:

```
random_seed: 1
search_method_names:
  - None
no_tune:
    no_tune: true
```

Otherwise you will need to define in `_configs/conf/stages/tune/[classifier_name].yaml` the following:
- `classifier_name`: the name of the classifier `TLClassifierEnum`
- `tuner_config`: a configuration of the tuning `TunerConfig`:
    - `hyperparam_tuning` the set of ray's hyperparameter to be tuned (alongside their search space).
    - `hyperparam_fixed` the set of ray's fixed hyperparameter that won't be tuned.
- `ray_config`: a configuration of ray `RayConfig` defined in `src/_utils/stages.py` - usually you only need to specify how many `cpu/gpu` your classifier will need.

### Config for train stage

Your training stage is necessary and cannot be skipped, therefore it will need to contain the following:
- `classifier_name`: the name of the classifier `TLClassifierEnum`
- `config`: the custom config dict of the classifier, if it is a `HuggingFaceClassifier` will need to follow the `HFConfig` present in `src/_classifiers/hf/base.py`.
- `tune_config`: a dict whose schema follows `TuneConfig` and should contain:
    - `search_method_name`: a string where you define which hyperparameter search method will be used to choose the best set of hparams, you have four options:
        - `None`: where the classifier's config should only be defined in the `train` stage - in other words `tune` stage should be skipped in this case.
        - The options defined by the base enumeration `TLTunerEnum` in `src/_utils/enumerations.py`.
    - `train_tune_option`: a string where you have five options defined in `TrainTuneOptionsEnum`:
        - `tune_configs_only`: where the classifier's config should only be defined in the `tune` stage.
        - `train_configs_only`: where the classifier's config should only be defined in the `train` stage - in other words `tune` stage should be skipped in this case.
        - `tune_overrides_train`: where the hparams defined in `train` stage will be overriden by the `tune` stage.
        - `train_overrides_tune`: where the hparams defined in `tune` stage will be overriden by the `train` stage.
        - `train_union_tune`: where the union of both hparams defined in `tune` and `train` stages will be taken as config for the classifier with no intersection allowed between the two sets defined in `tune` and `train`. 


## Defaults

Some pipeline configuration are usually not defined / overriden when adding a new classifier / dataset and are therefore defined in the `src/_configs/conf/config.yaml` with comments to explain each field what it does.