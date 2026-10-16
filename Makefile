# Variables (override on the command line, e.g. `make eval SCENARIO=credit_card_upgrade PROMPT=v3`)
#   SCENARIO  scenario pack (scenarios/<SCENARIO>/); default from config.yaml `scenario`
#   PROMPT    prompt version (prompts/<PROMPT>.md); v1/v2 are personal_loan only
#   REPEATS   simulated calls per persona (default config compare.repeats; use 1 for a quick run)
#   A, B      versions for `make compare`;  ARGS: extra sim args, e.g. ARGS="--n 3"
#   BEFORE, AFTER  bench JSON files for `make bench-compare`;  LABEL: bench run label
# (no trailing comments on variable lines: make would keep the spaces in the value)
PY := .venv/bin/python
CFG = $(PY) -c "from brain.config import load_config as l; c = l(); print($(1))" 2>/dev/null
SCENARIO ?= $(shell $(call CFG,c.scenario.id) || echo personal_loan)
PROMPT ?= v1
REPEATS ?= $(shell $(call CFG,c.compare.get('repeats', 1)) || echo 1)
A ?= v1
B ?= v2
ARGS ?=
LABEL ?= run
RESULTS := $(shell $(call CFG,c.results_dir) || echo results)
LATENCY := $(shell $(call CFG,c.voice['latency_log']) || echo logs/latency.jsonl)

.PHONY: help install run-voice sim postcall score report report-all eval compare bench \
	bench-compare demo-report demo-data test lint

help:
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/\t/'

install:  ## create .venv (Python 3.11) and install pinned deps
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

run-voice:  ## browser voice bot at http://localhost:7860/client (PROMPT=v3 SCENARIO=... to switch)
	PROMPT_VERSION=$(PROMPT) SCENARIO=$(SCENARIO) $(PY) -m voice.bot -t webrtc

sim:  ## text-mode simulator -> results/SCENARIO/PROMPT/transcripts (resumable)
	@$(PY) -m sim.simulate --scenario $(SCENARIO) --version $(PROMPT) --repeats $(REPEATS) $(ARGS); s=$$?; \
	if [ $$s -eq 3 ]; then echo "warning: some calls errored (excluded from scoring); re-run 'make sim' to retry them"; fi; \
	[ $$s -eq 0 ] || [ $$s -eq 3 ]

postcall:  ## CRM records for every transcript -> results/SCENARIO/PROMPT/postcall
	$(PY) -m brain.postcall --scenario $(SCENARIO) --version $(PROMPT)

score:  ## rules + grounding + LLM judge + cost -> scores/, costs/, summary.json (no re-simulation)
	$(PY) -m score.score --scenario $(SCENARIO) --version $(PROMPT)

report:  ## report.html for SCENARIO/PROMPT (+ voice latency for that version)
	$(PY) -m report.build_report --scenario $(SCENARIO) --version $(PROMPT) --results $(RESULTS) \
		--latency $(LATENCY) --out report.html

report-all:  ## report.html across every scenario with results for PROMPT (compares with A)
	$(PY) -m report.build_report --all-scenarios --version $(PROMPT) --compare-with $(A) \
		--results $(RESULTS) --latency $(LATENCY) --out report.html

eval: sim postcall score report  ## simulate -> post-call -> score -> report, one command

compare:  ## A vs B with 95% CIs -> results/SCENARIO/compare_A_B.json + report.html
	$(PY) -m ab.compare --a $(A) --b $(B) --scenario $(SCENARIO) --results $(RESULTS)
	$(PY) -m report.build_report --scenario $(SCENARIO) --version $(B) --compare-with $(A) \
		--results $(RESULTS) --latency $(LATENCY) --out report.html

bench:  ## LLM TTFB / first-clause / TTS TTFB benchmark -> logs/bench/LABEL-<ts>.json (real API calls)
	$(PY) -m bench.run --label $(LABEL) --version $(PROMPT) --scenario $(SCENARIO) $(ARGS)

bench-compare:  ## p50/p95 deltas between two bench runs: make bench-compare BEFORE=a.json AFTER=b.json
	$(PY) -m bench.compare $(BEFORE) $(AFTER)

demo-report:  ## build demo/report.html from bundled synthetic data (no API keys)
	$(PY) -m report.build_report --version v3 --all-scenarios --results demo/results \
		--compare-with v2 --latency demo/latency.jsonl --out demo/report.html --demo

demo-data:  ## regenerate the synthetic demo dataset
	$(PY) -m report.make_demo_data

test:  ## offline test suite
	$(PY) -m pytest -q

lint:  ## ruff check + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
