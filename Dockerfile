FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
LABEL name=yoshimidi
LABEL version=0.1.0
USER root
WORKDIR /app
# Apt dependencies.
RUN apt-get -y update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*
# Install s5cmd.
RUN S5CMD_DEB="$(mktemp)" && \
    wget 'https://github.com/peak/s5cmd/releases/download/v2.1.0/s5cmd_2.1.0_linux_amd64.deb' \
        --output-document "$S5CMD_DEB" && \
    dpkg -i "$S5CMD_DEB" && \
    rm "$S5CMD_DEB"
# Install Poetry.
ENV PYTHONIOENCODING=utf-8
ENV POETRY_CONFIG_DIR=/app/.poetry
ENV PATH="${PATH}:/opt/conda/bin"
RUN pip install poetry==1.4.1
RUN poetry config virtualenvs.create false
# Install Python dependencies via Poetry.
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-interaction
