BASE_PY ?= $(shell command -v python3.13 || command -v python3.12 || command -v python3)
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: install demo train eval test lint format compose-demo down data

install:  ## Create .venv (Python 3.13 or 3.12) and install pinned dependencies
	$(BASE_PY) -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

demo:  ## Train, promote, roll back, serve, replay traffic and trigger a drift alert (no Docker, no keys)
	@$(PY) scripts/demo.py 2>.demo.log || (tail -30 .demo.log; exit 1)

train:  ## Train and register a model (uses MLFLOW_TRACKING_URI from the environment)
	$(PY) scripts/train.py

eval:  ## Held-out precision, recall and confusion matrix -> results/metrics.md
	@$(PY) scripts/evaluate.py 2>/dev/null

test:  ## pytest with coverage
	$(PY) -m pytest

lint:  ## ruff lint and format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

data:  ## Re-download and verify the dataset from UCI
	$(PY) scripts/download_data.py

compose-demo:  ## Full stack in Docker, replay traffic, then show the logged row count
	docker compose up -d --build --wait
	$(PY) scripts/replay.py
	docker compose exec postgres sh -c 'psql -U $$POSTGRES_USER -d $$POSTGRES_DB -tc "select count(*) from prediction_logs"'

down:  ## Stop the Docker stack and delete its volumes
	docker compose down -v
