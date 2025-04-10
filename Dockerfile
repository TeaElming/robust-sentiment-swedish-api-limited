# Base image with CUDA 11.8 (compatible with PyTorch and most GPUs)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Copy environment file and install dependencies
COPY environment.yml .
RUN conda update -n base -c defaults conda && \
    conda env create -f environment.yml && \
    conda clean -afy

# Activate environment
SHELL ["conda", "run", "--no-capture-output", "-n", "vm-base", "/bin/bash", "-c"]

# Copy project files
COPY . /app
WORKDIR /app

# Expose the port your app runs on
EXPOSE 8002

# Start the FastAPI app
CMD ["conda", "run", "--no-capture-output", "-n", "vm-base", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
