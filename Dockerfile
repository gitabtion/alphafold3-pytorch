
ARG PYTORCH_TAG=2.3.0-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:${PYTORCH_TAG}

## Add System Dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive \
    && apt-get install --no-install-recommends -y \
        build-essential \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

## Install some Python dependencies
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && python -m pip install --no-cache-dir \
    pytest \
    requests \
    biopandas \
    omegaconf \
    wandb \
    pyscf \
    tensorboard \
    tensorboardX

## Change working directory
WORKDIR /app/alphafold

## Clone and install the package + requirements
ARG GIT_TAG=carsifold
RUN git clone https://github.com/gitabtion/alphafold3-pytorch.git . --branch ${GIT_TAG} \
    # && git checkout main \
    && python -m pip install .
