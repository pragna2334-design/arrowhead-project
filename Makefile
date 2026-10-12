# PROMPT: prompt version for sim/score/report/eval (prompts/<PROMPT>.md)
# A, B: versions for `make compare`; ARGS: extra sim args, e.g. ARGS="--n 3"
# (no trailing comments on these lines: make would keep the spaces in the value)
PY := .venv/bin/python
PROMPT ?= v1
A ?= v1
B ?= v2
ARGS ?=
# Paths come from config.yaml so changing paths.results_dir / voice.latency_log just works.
CFG = $(PY) -c "from brain.config import load_config as l; c = l(); print($(1))" 2>/dev/null
RESULTS := $(shell $(call CFG,c.results_dir) || echo results)
LATENCY := $(shell $(call CFG,c.voice['latency_log']) || echo logs/latency.jsonl)

.PHONY: help install run-voice sim postcall score report eval compare demo-report demo-data test lint

help:
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/\t/'

install:  ## create .venv (Python 3.11) and install pinned deps
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

run-voice:  ## browser voice bot at http://localhost:7860/client (PROMPT=v2 to switch prompt)
	PROMPT_VERSION=$(PROMPT) $(PY) -m voice.bot -t webrtc

sim:  ## text-mode simulator -> results/PROMPT/transcripts (resumable)
	@$(PY) -m sim.simulate --version $(PROMPT) $(ARGS); s=$$?; \
	if [ $$s -eq 3 ]; then echo "warning: some calls errored (excluded from scoring); re-run 'make sim' to retry them"; fi; \
	[ $$s -eq 0 ] || [ $$s -eq 3 ]

postcall:  ## CRM records for every transcript -> results/PROMPT/postcall
	$(PY) -m brain.postcall --version $(PROMPT)

score:  ## rules + LLM judge -> results/PROMPT/scores + summary.json (no re-simulation)
	$(PY) -m score.score --version $(PROMPT)

report:  ## report.html for PROMPT (+ voice latency for that version)
	$(PY) -m report.build_report --version $(PROMPT) --results $(RESULTS) --latency $(LATENCY) --out report.html

eval: sim postcall score report  ## simulate -> post-call -> score -> report, one command

compare:  ## A/B: results/compare_A_B.json + report.html with the comparison view
	$(PY) -m ab.compare --a $(A) --b $(B) --results $(RESULTS)
	$(PY) -m report.build_report --version $(B) --results $(RESULTS) --compare $(RESULTS)/compare_$(A)_$(B).json \
		--latency $(LATENCY) --out report.html

demo-report:  ## build demo/report.html from bundled synthetic data (no API keys)
	$(PY) -m report.build_report --version v1 --results demo/results \
		--compare demo/results/compare_v1_v2.json --latency demo/latency.jsonl \
		--out demo/report.html --demo

demo-data:  ## regenerate the synthetic demo dataset
	$(PY) -m report.make_demo_data

test:  ## offline test suite
	$(PY) -m pytest -q

lint:  ## ruff check + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
