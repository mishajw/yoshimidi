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
vast-ssh: vast-rsync
	HOST=$$( \
		vastai show instances --raw \
		| jq -r '.[] | .ssh_host') && \
	PORT=$$( \
		vastai show instances --raw \
		| jq -r '.[] | .ssh_port') && \
	ssh root@$$HOST -p $$PORT bash -c "cd /app && make $(CMD)"

.PHONY: vast-rsync
vast-rsync:
	HOST=$$( \
		vastai show instances --raw \
		| jq -r '.[] | .public_ipaddr') && \
	PORT=$$( \
		vastai show instances --raw \
		| jq -r '.[] | .direct_port_start') && \
	rsync -r \
		-e "ssh -p $$PORT" \
		--filter=':- .gitignore' \
		--filter='- .git' \
		. \
		root@$$HOST:/app && \
	rsync -r \
		-e "ssh -p $$PORT" \
		.env root@$$HOST:/app/.env
