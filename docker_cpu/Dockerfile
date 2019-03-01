# dockerfile containing a conda environment configured to run the tests inside this project
# or to build the python package with setup.py
# the container doesn't contain the library itself
FROM ubuntu:18.04
MAINTAINER Mathieu Lamarre <mathieu.lamarre@gmail.com>

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    pkg-config \
    libcurl4-openssl-dev \
    zlib1g-dev \
    nano && \
    apt-get autoremove && \
    apt-get clean

# Using Miniconda to get  3.6.y to get Numpy linked with Intel MKL
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/anaconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:$PATH
RUN conda update -n base conda && conda create -y -n cmhm python=3.6 numpy future opencv scipy pillow
RUN /bin/bash -c "source /opt/conda/bin/activate cmhm && pip install --upgrade pip"
RUN /bin/bash -c "source /opt/conda/bin/activate cmhm && conda install -y -c menpo cyvlfeat vlfeat menpo --no-deps"
RUN /bin/bash -c "source /opt/conda/bin/activate cmhm && conda install -y -c conda-forge matplotlib jupyter jupyterlab tqdm tensorflow nodejs"

# Facultative : jupyterlab
RUN /bin/bash -c "source /opt/conda/bin/activate cmhm && jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
    jupyter labextension install ipyvolume && \
    jupyter labextension install jupyter-threejs"

# When running with --user 1000
RUN groupadd appuser -g 1000 && useradd -r -u 1000 appuser -g appuser

# Example to run unit-tests
# First download models from https://drive.google.com/open?id=1DKTeRlJjyo_tD1EluDjYLhtKFPJ9vIVd
# Unzip in the same folder as this file
# docker build --network=host -t cmhm3dfa:latest ./docker_cpu
# docker run --network=host -v $PWD:/root/ --rm -ti cmhm3dfa:latest /bin/bash -c "source /opt/conda/bin/activate cmhm && cd ~ && python tests/test_cmhm3dfa.py"
