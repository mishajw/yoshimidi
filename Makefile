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
	poetry run mypy ./yoshimidi ./tests

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
