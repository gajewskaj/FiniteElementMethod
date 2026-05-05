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

RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# cuDSS libraries are installed under a versioned directory; add it to the loader search path.
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcudss/12:$LD_LIBRARY_PATH

CMD ["bash"]