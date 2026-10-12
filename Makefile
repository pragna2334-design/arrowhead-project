# PROMPT: prompt version for sim/score/report/eval (prompts/<PROMPT>.md)
# A, B: versions for `make compare`; ARGS: extra sim args, e.g. ARGS="--n 3"
# (no trailing comments on these lines: make would keep the spaces in the value)
PY := .venv/bin/python
PROMPT ?= v1
A ?= v1
B ?= v2
ARGS ?=
RESULTS := results

.PHONY: help install run-voice sim postcall score report eval compare demo-report demo-data test lint

help:
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/\t/'

install:  ## create .venv (Python 3.11) and install pinned deps
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

run-voice:  ## browser voice bot at http://localhost:7860/client (PROMPT=v2 to switch prompt)
	PROMPT_VERSION=$(PROMPT) $(PY) -m voice.bot -t webrtc

sim:  ## text-mode simulator -> results/$(PROMPT)/transcripts (resumable)
	@$(PY) -m sim.simulate --version $(PROMPT) $(ARGS); s=$$?; \
	if [ $$s -eq 1 ]; then echo "warning: some calls errored; re-run 'make sim' to resume them"; fi; \
	[ $$s -ne 2 ]

postcall:  ## CRM records for every transcript -> results/$(PROMPT)/postcall
	$(PY) -m brain.postcall --version $(PROMPT)

score:  ## rules + LLM judge -> results/$(PROMPT)/scores + summary.json (no re-simulation)
	$(PY) -m score.score --version $(PROMPT)

report:  ## report.html for $(PROMPT) (+ latency from logs/latency.jsonl)
	$(PY) -m report.build_report --version $(PROMPT) --latency logs/latency.jsonl --out report.html

eval: sim postcall score report  ## simulate -> post-call -> score -> report, one command

compare:  ## A/B: results/compare_$(A)_$(B).json + report.html with the comparison view
	$(PY) -m ab.compare --a $(A) --b $(B) --results $(RESULTS)
	$(PY) -m report.build_report --version $(B) --compare $(RESULTS)/compare_$(A)_$(B).json \
		--latency logs/latency.jsonl --out report.html

demo-report:  ## build demo/report.html from bundled synthetic data (no API keys)
	$(PY) -m report.build_report --version v1 --results demo/results \
		--compare demo/results/compare_v1_v2.json --latency demo/latency.jsonl \
		--out demo/report.html --demo

demo-data:  ## regenerate the synthetic demo dataset
	$(PY) -m report.make_demo_data

test:  ## offline test suite
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
