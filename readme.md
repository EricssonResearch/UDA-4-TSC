# Unsupervised Domain Adaptation for Time Series Classification (UDA-4-TSC)

This repository contains the official implementation of the benchmark titled "DEEP UNSUPERVISED DOMAIN ADAPTATION FOR TIME SERIES CLASSIFICATION: A BENCHMARK". 

## Contributions 

For contributing follow the `contribution.md` guide.

## Requirements

Make sure you have an env with `python==3.10` and run the following: 

```
pip install -r requirements.txt 
```

## Run

Make sure you are inside `src/`, then to run a short pipeline of `CoDATS` model on `har` dataset with `source=12` and `target=16` follow the two commands:

### Generate the config with hydra

```
python3 -m _utils.generate_conf 'stages/preprocess=har' 'stages/train=CoDATS' 'utils.idxExperiment=1' 'stages.tune.tuner_config.address=null' 'stages.tune.ray_config.resume=false' 'stages.tune.ray_config.num_samples=1' 'global.train_time_limit=15' 'global.tune_time_limit=30' '++stages.tune.tuner_config.hyperparam_fixed.device=cpu' 'stages.tune.ray_config.resources_per_trial.gpu=0'
```

### Run the pipeline 

```
bash run.sh
```

### View result 

```
cat output/results/dataset_name\=har/source\=12/target\=16/classifier_name\=CoDATS/metrics.json
```

### Docker

To build the docker image: 

```
docker build -t uda-4-tsc:latest .
```

To run the docker image:

```
docker run --name uda-4-tsc --network host --entrypoint bash -idt -v /path/to/cloned/uda-4-tsc:/tmp  uda-4-tsc:latest
```

Now you have the necessary env to run the code.

Alternatively you can use the already published docker image on docker hub: TODO