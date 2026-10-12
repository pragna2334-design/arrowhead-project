PY := .venv/bin/python

.PHONY: install run-voice sim score report eval test lint

install:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

run-voice:  ## browser voice bot at http://localhost:7860/client
	$(PY) -m voice.bot -t webrtc

sim score report eval:  ## Phase 2/3
	@echo "TODO: '$@' is implemented in Phase 2/3" && exit 1

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .
