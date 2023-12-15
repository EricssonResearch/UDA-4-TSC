FROM ubuntu:22.04

RUN apt update -y
RUN apt install -y python3
RUN apt install -y python3-pip 

COPY requirements.txt .
RUN pip install -r requirements.txt

USER 1000

# set the cache dir for transformers 
ENV TRANSFORMERS_CACHE "/tmp/cache/transformers"

# set the cache dir for hf datasets 
ENV HF_DATASETS_CACHE "/tmp/cache/datasets"

# working directory 
WORKDIR "/tmp"

# entrypoint
ENTRYPOINT ["bash", "echo", "override this when you run"]