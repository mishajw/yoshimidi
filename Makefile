# Development
# ===========

.PHONY: install
install:
	poetry install

.PHONY: precommit
precommit:
	poetry run pre-commit run --all-files

.PHONY: mypy
mypy:
	poetry run mypy .

.PHONY: test
test:
	poetry run pytest

.PHONY: du
du:
	@echo "> out"
	@du -sh out/*
	@echo "> dataset"
	@du -sh out/dataset/*
	@echo "> checkpoints"
	@du -sh out/checkpoints/*

# Dataset prep
# ============

.PHONY: parse
parse:
	poetry run python yoshimidi/clis/parse_midi_files.py

.PHONY: tokenize
tokenize:
	poetry run python yoshimidi/clis/tokenize_midi_dataset.py

.PHONY: train
train:
	poetry run python yoshimidi/clis/train.py configs/train.toml

# S3
# ==

.PHONY: s3-download
s3-download:
	env \
		$$(cat .env | xargs) \
		s5cmd sync 's3://yoshimidi/datasets/2023-07-29/*_0000.npy' out/dataset_tokenized

# Vast.ai
# =======

.PHONY: vast-ssh
vast-ssh:
	poetry run python youshimidi/clis/vastai.py ssh

.PHONY: vast-make
vast-make: vast-rsync
	poetry run python youshimidi/clis/vastai.py make $(CMD)

.PHONY: vast-rsync
vast-rsync:
	poetry run python youshimidi/clis/vastai.py rsync
