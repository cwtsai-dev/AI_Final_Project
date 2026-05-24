# SmartFolio — heuristic-guided IRL portfolio optimization (GPU build)
#
# Requires NVIDIA Container Toolkit on the host. Build & run:
#   docker build -t smartfolio .
#   docker run --rm --gpus all smartfolio
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.8 + build tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3.8-distutils python3-pip \
        build-essential git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.8 /usr/bin/python

WORKDIR /app

# gym 0.21 only installs/imports cleanly with older pip/setuptools/wheel
RUN python -m pip install --upgrade "pip==23.0.1" "setuptools==65.5.0" "wheel==0.38.4"

# 1) PyTorch with CUDA 11.3 (matches the 11.3.1 base image)
RUN pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 2) PyG compiled extensions, matched exactly to torch 1.12.1 + cu113
RUN pip install torch-scatter==2.0.9 torch-sparse==0.6.15 \
        -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

# 3) Remaining Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4) Project source + data
COPY . .

# main.py auto-selects cuda:0 when torch.cuda.is_available(); reads ./dataset/...
CMD ["python", "main.py"]
