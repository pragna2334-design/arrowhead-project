# Session handoff

Last updated: 2026-10-16. Read this first in a new session, then `README.md`, `DECISIONS.md` and `AUDIT.md`.

## Where things stand

- **Repo:** `pragna2334-design/arrowhead-project`. All work is on `main`, and everything is pushed. GitHub's default branch should be switched to `main` in Settings → General; the old `claude/sharp-cannon-j8ivct` branch can be deleted.
- **Tests and CI:** 572 offline tests, all passing. Lint is clean. CI runs ruff, pytest and `make demo-report`.
- **v1/v2 prompts are frozen.** Neither file has changed since `761172a`. Tests pin their md5 hashes and their rendered output, and both refuse non-loan scenarios. **Do not edit them.**
- **LLM:** DeepSeek only. Agent `deepseek-flash`; judge `deepseek-pro`, which is a guess, so fix `brain.judge_model` if the judge errors.
- **User's machine:** WSL Ubuntu at `~/arrowhead-project`. `.env` has DEEPGRAM, SARVAM and DEEPSEEK keys. Voice calls already worked in the earlier round.

### Built this round (round 3)

- **Scenario packs** (`scenarios/<id>/scenario.yaml` + `personas.yaml`):
  - `personal_loan` (25 personas)
  - `health_insurance_renewal` (15)
  - `credit_card_upgrade` (15)
  - Each pack has 5 red-team personas. Results are stored under `results/<scenario>/<version>/`.
- **Prompt v3:** scenario-generic and built on v2's hardening. It opens replies with short acknowledgements and takes figures only from tools. The EMI text appears only where EMI is enabled.
- **Grounding:**
  - `brain/finance.py`: `calculate_emi`, `get_product_terms`.
  - `score/grounding.py`: the hallucinated-figure rule.
  - Scenario forbidden claims are checked in `score/scenario_rules.py`.
- **Latency:**
  - `brain.thinking` is validated and logged at startup.
  - Clause-level streaming into TTS (`voice/chunking.py`, `voice/aggregation.py`).
  - `make bench` and `make bench-compare` (`bench/`).
- **Cost:** `score/cost.py` gives production vs eval cost and cost per booked callback, priced from `config.yaml → prices`. All prices are placeholders.
- **Rigor:** `ab/stats.py` computes Wilson and Newcombe intervals for rates and a seeded bootstrap for means, and flags differences that aren't significant (n.s.).
- **Report:** a tab per scenario, plus cost and hallucinated-figure sections and CI-based comparisons.
- **Demo:** `make demo-report` shows every section from synthetic data.
- **Independent review:** 4 major and 8 minor findings, all fixed (see AUDIT.md, round 3).

## Needs local verification (cannot be done in the cloud sandbox)

1. **Model ids:** `deepseek-flash` and `deepseek-pro` are both unverified.
2. **Thinking:** whether DeepSeek really stops reasoning with `thinking: disabled`. `make bench` reports `reasoning_chunks`; anything above 0 means it doesn't.
3. **Sarvam:**
   - which `min_buffer_size` values are accepted (currently 50; try 10–20);
   - whether a short first clause really plays early;
   - how clause chunks sound.
4. **Grounding and rules on real transcripts:** false-positive and false-negative rates, especially for unattributed customer echoes and non-offer numbers of 100 or more.
5. **v3 on the insurance and card packs:** how it behaves, and how realistic the new personas are.
6. **Prices:** check `config.yaml → prices` against your invoices.
7. **Report in a browser:** the radio-button tabs, dark mode and phone width.
8. **Voice checks (still open from the last round):**
   - "rate zyada hai"
   - a 9 pm callback is refused
   - "haan" does not interrupt the bot
   - "dobara call mat karna" confirms DNC and hangs up

## Ordered checklist to fill the README results table

Run these in order from `~/arrowhead-project`, after `git pull && make install`.

```bash
# 0. sanity (no keys needed)
make test && make demo-report

# 1. latency: bench before/after (real API calls; ~15 utterances x 3 repeats)
make bench LABEL=before PROMPT=v2 ARGS="--aggregation sentence"
make bench LABEL=after  PROMPT=v3
make bench-compare BEFORE=$(ls -t logs/bench/before-*.json | head -1) AFTER=$(ls -t logs/bench/after-*.json | head -1)

# 2. live voice, after: a few calls on v3 (logs/latency.jsonl gets prompt_version v3)
make run-voice PROMPT=v3

# 3. smoke runs first (cheap; catch model-id or key problems before the long runs)
make eval PROMPT=v1 REPEATS=1 ARGS="--n 3"
make eval PROMPT=v3 REPEATS=1 ARGS="--n 3" SCENARIO=health_insurance_renewal

# 4. full personal_loan evals (25 personas x 3 repeats each; hours at 8 rpm)
make eval PROMPT=v1
make eval PROMPT=v2
make eval PROMPT=v3

# 5. comparisons with 95% CIs (report.html is rewritten each time; open it after each)
make compare A=v1 B=v2
make compare A=v2 B=v3

# 6. other scenarios on v3
make eval PROMPT=v3 SCENARIO=health_insurance_renewal
make eval PROMPT=v3 SCENARIO=credit_card_upgrade

# 7. one report across all scenarios (v3, compared with v2 where it exists)
make report-all PROMPT=v3 A=v2        # open report.html (WSL: explorer.exe report.html)
```

Then fill in:
- README → Results: the personal_loan v1/v2/v3 table, the Scenarios-on-v3 table and the Latency table.
- The `_X_/_Y_/_A_/_B_/_C_/_N_` placeholders in the README resume summary.

Report a delta only where the report marks it significant, not n.s.

## Useful facts for the next session

- **Pipecat is pinned at 1.12.0.** Verify APIs against `.venv/lib/python3.11/site-packages/pipecat`; the docs site is unreachable from the sandbox. Clause streaming swaps `tts._text_aggregator` the same way Pipecat's Cartesia/Rime services do, and fails loudly at startup if Pipecat internals change.
- **Where the code lives:**
  - Contracts: `schemas/` (frozen pydantic).
  - Scenario loading: `brain/scenario.py`; config is `load_config(prompt_version=, scenario=)` plus the env vars `PROMPT_VERSION` and `SCENARIO`.
  - Shared tools: `brain/tools.py` (`tool_specs(scenario)`, `execute_tool(..., scenario=)`).
  - Usage capture: `brain/usage.py`.
- **Exit codes:**
  - The simulator exits 0 when everything ran, 2 on a setup error (bad key, scenario/prompt pairing, personas) and 3 when only some calls errored. Errored calls are excluded from scoring.
  - `score` exits 2 when there are no transcripts.
- **Known limitations:** see README → Limitations. They include STT billing, unrecorded voice-call token usage, correlated repeats, the keyword-based grammar, and Devanagari disclosure for non-loan packs.
- **Commits** end with the session's Co-Authored-By and Claude-Session trailers. Push to `main`.
