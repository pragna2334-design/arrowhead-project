# Decisions (one line each: decision — reason)

Items marked `VERIFY` need a live run with real keys. AUDIT.md lists the bugs fixed after Phase 1.

## Stack and models
- Pinned `pipecat-ai==1.12.0`, with every import checked against its installed source — the docs site was unreachable, so the source was the authority.
- Python 3.11 via `uv` — spec requirement; the sandbox default is 3.13.
- DeepSeek is the only LLM provider (agent, simulated customer, judge) — the user has a DeepSeek key and no Gemini key.
- Agent `deepseek-flash`, judge `deepseek-pro`, pinned in config, never the library default — `deepseek-flash` is Pipecat 1.12.0's name for V4.1 Flash; `deepseek-pro` is a guess. `VERIFY` both ids.
- The judge must differ from the agent, checked at config load, at judge creation and per transcript — reduces self-grading bias.
- DeepSeek thinking is disabled for the agent and customer (`extra_body.thinking`) — V4 otherwise reasons before every answer, which hurts voice TTFB.
- Text mode uses the `openai` SDK at Pipecat's DeepSeek base URL — same endpoint and model settings as voice; only the loop code differs (README limitation).
- Ruff pinned to 0.16.8 — its default rule set changes between versions, so CI and local runs would otherwise disagree.

## Voice
- Deepgram `nova-3-general` with `language: "multi"` passed as a raw string — Hinglish code-switching. `VERIFY`; fall back to `"hi"` or `stt.provider: sarvam`.
- Sarvam `bulbul:v3`, `hi-IN`, mixed text sent as-is — Sarvam has no Hinglish code.
- Female agent "Priya" (voice `priya`, feminine verb forms via `brain.agent_gender`) — the first live test showed the default `shubh` voice is male.
- Hindi in Roman script by default (`brain.script`) — simplest for the LLM and for Sarvam. `VERIFY` by listening.
- Closing lines are fixed text in the brain, spoken via `TTSSpeakFrame` before `EndWorkerFrame` — the DNC confirmation is always said and a call never ends in silence (AUDIT #1).
- Barge-in needs at least 2 words while the bot is speaking (`MinWordsUserTurnStartStrategy`) — "haan" / "hmm" / noise must not cut the agent off. `interrupt_min_words: 0` restores Pipecat's VAD default.
- The customer is muted until the first bot utterance completes and during tool calls — the disclosure is always heard in full, and a barge-in can't cancel consent or hang-up.
- Mood is tagged on the context message (MoodTagger processor), not on the transcription frame — the browser transcript stays clean while the LLM sees the same tag as in text mode. The in-place edit persists (verified in source: `get_messages()` returns the live list).
- Voice transcripts are rebuilt from the LLM context at call end, with synthetic timestamps — Pipecat context messages carry no times; it's simple and needs no extra processor.
- Post-call records are written in a `finally` after `runner.run()` — covers both bot hang-up and customer disconnect.
- Opening cue is a user-role message in both modes — DeepSeek has no `developer` role.
- `pipeline_ms` = observer contributions minus endpointing_wait, transcription, first_request and output_transport — this matches "STT final → first TTS audio". `VERIFY` the keys in a live call.

## Brain and prompts
- Prompts live in `prompts/<version>.md`, selected by `brain.prompt_version` or `PROMPT_VERSION` — makes A/B testing a config switch.
- v1 keeps the Phase 1 sales and compliance wording; only its Tools section changed to match the new mechanics — v1 stays a fair baseline (AUDIT #16).
- v2 adds an instruction hierarchy, privacy rules, handling for difficult callers, explicit DNC-vs-decline, Hinglish→24h time conversion and a fixed opening shape — these target the predicted red-team and rule failures.
- `book_callback(day, time)` uses 24-hour HH:MM checked against `compliance.callback_hours: [10, 19)` — out-of-hours callbacks are both prevented and auditable. The hours are a fictional policy; adjust per your compliance team.
- `mark_do_not_call` also ends the call, and no tool runs after the call has ended — stopping immediately is the compliance requirement.
- Mood is a word-list classifier (calm/annoyed/confused/interested) with whole-word matching for Latin words — no latency or cost, deterministic, identical in voice and text mode.
- Spoofed `[customer_mood:]` tags are stripped from customer text — a customer must not steer the agent's tone.

## Simulator
- 25 personas (20 normal and 5 red-team, one per attack type), all fictional, nearly all named "Rahul Verma"; `wrong_person` plays someone else — matches the configured customer.
- One shared limiter spaces requests evenly (60/rpm s) across agent and customer — steady traffic is less likely to hit 429s than bursts.
- Calls run one at a time — the rate limit is the bottleneck anyway, and it keeps things simple.
- Resume is decided by the transcripts on disk (valid and not `error`); `sim_state.json` is only a log — a single source of truth.
- Errors still write the transcript with `stop_reason: error`, the simulator exits 3, and `make eval` continues — no lost evidence; re-running resumes the failed calls. Exit 2 (setup) or a crash stops the pipeline.
- Errored calls are excluded from scoring, post-call and the summary, and listed in `errored_calls` — a crashed call says nothing about the agent.
- Post-call records and scores are rebuilt when the transcript is newer — re-simulated calls never keep stale results.
- The openai client is built with `max_retries=0`; our own loops retry 429/5xx/network errors behind the rate limiter — no hidden bursts.
- The customer LLM replies are capped at 150 tokens, it hangs up with a bare `<hangup>`, and the hang-up is not recorded as a turn — short spoken replies and clean transcripts.

## Scoring
- Compliance is decided only by deterministic rules; the judge scores quality and never overrides a rule — reproducible and testable offline.
- The judge sees the redacted transcript, persona and a public policy summary, never the agent's system prompt — it grades behaviour, not adherence to the prompt's wording.
- Judge: anchored 1-5 rubric, temperature 0, JSON output, thinking disabled — repeatable scores.
- The judge is best-effort: retries, then `judge_error`, and a re-run re-judges only the failures — an outage never blocks scoring.
- Only the first stop-calling request is checked for DNC — honouring a second request is still a failure.
- A rate quoted with someone else's name in the same sentence ("aapke bank ne 10.5% bola") is not counted — when unsure, the quote counts against the agent.
- Guarantee words are cleared by a negation within a few words; "reject nahi hoga" is still flagged — "won't be rejected" is a guarantee.
- Outcome: dnc > callback_booked > declined (ended, including wrong_person) > incomplete (max turns, hang-up, error) — matches the four outcomes in the spec.
- `red_team_reason` was added to `CallScore` — it keeps the judge's own explanation for the report.

## Post-call and PII
- Post-call records are deterministic (rules and word lists, no LLM) — free and repeatable.
- The summary line is built from a template without customer text, then redacted — PII can't leak through it.
- Redaction order is email → Aadhaar → PAN → phone, with phones requiring a leading 6-9 and standing alone — avoids false hits on money, rates, times and dates.
- Redaction happens at write time (transcripts, post-call records) and in logs via a loguru patcher — raw PII never reaches disk or the console.

## Report and demo
- One static HTML file with inline CSS/SVG and no JavaScript — works offline and from disk, with nothing to load from outside.
- Percentiles use linear interpolation (numpy type 7) written by hand; greetings are excluded — no numpy dependency, and greetings are timed from connect, not from speech.
- In the A/B view, a lower DNC share is not treated as better — honouring stop requests is correct behaviour.
- The per-persona A/B outcome is the most common outcome across repeats; violations are summed — stable with `--repeats`.
- Voice latency in a report is filtered to that prompt version — A/B latency compares like with like.
- The demo report is pinned to the data timestamp — the committed file doesn't change on rebuild.
- `demo/` is synthetic data, deterministic and clearly labelled (`demo-` ids, `-demo` models, `[demo]` text, a banner) — so anyone can see the report without keys, and it can't be mistaken for real results.

## Round 3: scenarios, grounding, latency, cost, rigor

### Contracts and prompts
- Use-case settings moved from config.yaml into scenario packs (`scenarios/<id>/scenario.yaml`, validated by `schemas/scenario.py`); config.yaml keeps the global settings — adding a use case is a YAML change, not a code change.
- Results layout is `results/<scenario>/<version>/` — scenarios never mix; no real results existed yet, so nothing needed migrating.
- prompts/v1.md and v2.md are frozen: tests pin their file hashes and their rendered text (golden files captured before the refactor), and they refuse non-loan scenarios — a clean v1-vs-v2 comparison.
- Prompt v3 = v2's hardening + scenario-generic placeholders + short acknowledgements + "figures only from tools"; it contains no offer numbers at all — the grounding rule then checks the model, not the prompt.
- Grounding tools (`get_product_terms`, `calculate_emi`) are offered to every prompt version — the same tool set for v1/v2/v3 keeps the A/B fair; v1/v2 simply aren't told to use them.
- `calculate_emi` is only offered when the scenario has `emi_enabled` (lending) — no EMI for insurance or cards.
- Token usage is captured on every agent and customer response and stored on the transcript — enables cost per call without extra API calls.
- Personal-loan values (company, band 11.5-18%, Rs 5,00,000, 12-60 months, 10:00-19:00, Rahul Verma) are pinned by tests — the scenario refactor can't silently change v1/v2 behaviour.

### Grounded numbers
- EMI uses reducing balance with Decimal and half-up rounding to the rupee; total payable = rounded EMI × months — exact and reproducible (1 lakh @ 12% for 12 months = Rs 8,885).
- `calculate_emi` returns an error (never raises) for rates outside the band, amounts above the maximum or tenures outside the range — the tool itself can't quote off-band terms.
- A figure is grounded if it matches any successful tool result in the call or a scenario value (unit-aware: 5 lakh = 500000, 5 saal = 60 months; pct ±0.05, rupees ±0.5, EMI ±1) — spec: "match a tool result or config value".
- An echo of the customer's number is allowed only if the agent attributes it in the same sentence AND the customer said it earlier — stops invented competitor rates.
- Times, dates, minutes, list numbers, unitless numbers < 100 and the fixed closing line are ignored — they aren't offer figures.
- The rate rule ignores percentages equal to a scenario fee or figure (2% reward rate, 20% no-claim bonus) unless a rate word is next to them; with no rate band the rule passes — no false rate violations outside lending.

### Scenarios
- Three packs, one per Arrowhead segment: personal_loan (25 personas), health_insurance_renewal and credit_card_upgrade (15 each; 5 red-team per pack) — lending, insurance, cards.
- Forbidden claims are scenario regexes (e.g. "no waiting period", "lifetime free", "claim guaranteed"), negation-aware, but a negation that is part of the claim itself never clears it — "no waiting period" is the claim.
- personal_loan also got forbidden claims (no processing fee, no documents, guaranteed same-day disbursal) — this applies equally to v1 and v2, so the comparison stays fair.
- Objection ids come from the scenario; the hand-tuned loan detectors are merged in for shared ids — loan detection quality is unchanged.

### Latency
- `brain.thinking` is validated (enabled|disabled only), and the bot logs the mode at startup — a typo can't silently turn reasoning on.
- Clause streaming replaces Pipecat's sentence aggregator inside the TTS service (the same pattern Pipecat's Cartesia/Rime services use); it fails loudly if Pipecat internals change — keeps the latency observer's aggregation metric.
- Clause rules: boundaries , ; : . ? ! । ॥ and spaced dashes; the first clause at 6+ characters, later ones at 20+; never split numbers/times/abbreviations — fast acknowledgement without choppy prosody.
- Sarvam `min_buffer_size` stays at Pipecat's 50 until verified live — a lower value may speed the first clause but could be rejected by the API.
- `make bench --aggregation sentence` uses Pipecat's own aggregator — an honest "before" number; TTS connect time is reported separately from TTS TTFB.

### Cost and statistics
- Production cost (agent LLM + STT + TTS) is separated from eval cost (+ simulated customer + judge) — only production cost says anything about unit economics.
- Text-mode audio is estimated from text (14 chars/s TTS, 2.5 words/s STT); unknown prices count as 0 with a warning — the report never crashes on a missing price; all prices are placeholders to verify.
- Costs are re-priced on every scoring run — price-table edits take effect without re-simulating or re-judging.
- Rates use Wilson intervals, and differences use Newcombe's hybrid score; means use a seeded percentile bootstrap (2,000) — standard, stdlib-only, deterministic.
- "Significant" = 95% CI of the difference excludes 0; everything else is n.s. and shown neutrally — no over-claiming.
- Default `REPEATS` = config `compare.repeats` (3) — comparisons need several samples per persona; `REPEATS=1` for smoke runs.
- Known caveats are stated in the report notes: correlated repeats, no multiple-comparison correction.

### Report
- Scenario tabs use radio inputs + CSS (no JavaScript); with one scenario the old flat layout is kept — works offline and from disk.
- Latency is shown once per report (records carry no scenario).

### Round 3 review decisions
- Red-team injection and guarantee-bait verdicts include grounding and scenario-claim rules — those packs aim their attacks exactly there.
- Rate vs fee is decided by the context before the number first (label word → fee; rate cue → rate), then the context after — "interest rate 9%, processing fee nahi" stays a rate.
- GST/tax percentages and short contact/spend timeframes ("2 din mein call") are not offer figures; other timeframes ("disbursal 3 din mein") still are.
- Forbidden claims exempt conditional, hedged and already-served sentences; trade-off: an invented conditional promise ("agar aaj lein to lifetime free") is not caught by this rule.
- `make score` refuses to write an empty summary — an empty report would hide a missing simulation.
- Tests clear SCENARIO/PROMPT_VERSION/CONFIG_PATH — the suite must not depend on the user's shell.

## Not verified (needs live keys or a browser)
DeepSeek model ids and JSON mode alongside the thinking setting; whether `reasoning_content` appears with thinking disabled (`make bench` reports it); Sarvam `min_buffer_size` limits and clause prosody; prices; grounding false-positive rate on real v3 transcripts; how v3 handles the insurance and card packs; Deepgram `multi` on Hinglish; Sarvam pronunciation; the goodbye audio playing fully before hang-up (and the closing line reaching the voice transcript); real latency numbers; how the simulated customers behave; the report in a real browser.
