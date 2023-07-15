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


# Dataset prep
# ============

.PHONY: parse
parse:
	poetry run \
		python yoshimidi/clis/parse_midi_files.py \
		out/dataset

.PHONY: tokenize
tokenize:
	poetry run \
		python yoshimidi/clis/tokenize_midi_dataset.py \
		out/dataset/dataset_parsed.jsonl \
		out/dataset_tokenized

.PHONY: train
train:
	poetry run \
		python yoshimidi/clis/train.py \
		out/dataset_tokenized
