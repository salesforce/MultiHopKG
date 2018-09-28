FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         locales \
         cmake \
         git \
         curl \
         vim \
         unzip \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         libfreetype6-dev \
         libxft-dev &&\
     rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda90 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN conda install -c pytorch pytorch=0.4.1 cuda90

RUN pip install --upgrade pip
RUN pip install tqdm==4.9.0 &&\
     pip install matplotlib==2.1.2

WORKDIR /workspace
RUN mkdir MultiHopKG
WORKDIR  MultiHopKG
