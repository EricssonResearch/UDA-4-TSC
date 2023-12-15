#/bin/bash
set -e
python3 -m _pipeline.preprocess
python3 -m _pipeline.tune
python3 -m _pipeline.train
python3 -m _pipeline.predict
python3 -m _pipeline.score