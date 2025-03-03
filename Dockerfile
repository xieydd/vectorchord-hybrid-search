FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

RUN set -eux; \
    apt-get update && apt-get install -y \
    curl vim wget python3-pip; \
    apt-get autoremove -y; \
    apt-get clean -y; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt