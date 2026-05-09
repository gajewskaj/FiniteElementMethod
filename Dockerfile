FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    build-essential \
    gmsh \
    libcudss0-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

RUN apt update
RUN apt install -y --no-install-recommends gnupg
RUN . /etc/os-release \
    && UBUNTU_VERSION="$(echo "$VERSION_ID" | tr -d .)" \
    && echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu${UBUNTU_VERSION}/$(dpkg --print-architecture) /" \
        | tee /etc/apt/sources.list.d/nvidia-devtools.list
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt update
RUN apt install -y --no-install-recommends nsight-systems-cli

RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# cuDSS libraries are installed under a versioned directory; add it to the loader search path.
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcudss/12:$LD_LIBRARY_PATH

CMD ["bash"]