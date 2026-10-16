# Phase 1 audit

I reviewed the Phase 1 code against the installed Pipecat 1.12.0 source. Each row is issue → fix. "Live" means it still needs a real voice call to confirm.

| # | Area | Issue | Fix |
|---|---|---|---|
| 1 | Hang-up race | Terminal tools relied on the LLM saying goodbye in the same response as the tool call. If the model called `end_call`/`mark_do_not_call` with no text, the call ended in silence, so the do-not-call request was never confirmed to the customer. | A fixed closing line from `brain.tools.closing_line()` is spoken with `TTSSpeakFrame` before `EndWorkerFrame`. Text mode appends the same line. The DNC confirmation and callback time are now always said. |
| 2 | Hang-up ordering | It was unclear whether pushing `EndWorkerFrame` from a tool handler would cut off queued audio. | Checked in source: a downstream `EndWorkerFrame` reaches the worker sink, is re-queued upstream, and becomes an `EndFrame` queued *behind* the speech. The output transport drains queued audio before stopping. The order is documented in `voice/bot.py`. Still needs a live call to confirm. |
| 3 | Barge-in vs tools | `cancel_on_interruption` defaults to True. Customer speech during `record_consent`/`book_callback`/`end_call` could cancel the handler, losing consent or leaving the call open. | `FunctionCallUserMuteStrategy` mutes customer input while a tool runs. |
| 4 | Disclosure | An early "hello?" could interrupt the first sentence, so the AI + lender disclosure was never fully heard. | `MuteUntilFirstBotCompleteUserMuteStrategy`: the greeting always plays in full. |
| 5 | Barge-in tuning | With VAD-triggered barge-in, "haan", "hmm", a cough or background noise cut the agent mid-sentence. | `MinWordsUserTurnStartStrategy(min_words=2)` applies while the bot is speaking; one word is enough when it is silent. Configurable with `voice.interrupt_min_words` (0 restores Pipecat's VAD default). |
| 6 | Tool state | Tools still ran after the call ended, e.g. `record_consent` after `mark_do_not_call` in the same response. A duplicate `end_call` could speak twice and push two end frames. | `execute_tool` rejects every tool once `state.ended`. The voice handler only speaks and ends once (`was_ended` guard). |
| 7 | Callback hours | `book_callback(time)` took free text ("tomorrow 9 pm"), so out-of-hours callbacks could be neither prevented nor audited. | Structured `day` + `time` (HH:MM, 24h), checked against `compliance.callback_hours` ([10, 19)). An out-of-window time returns an error to the model so it offers another time. |
| 8 | Auditability | Voice calls left nothing on disk except one log line. | A redacted transcript and post-call CRM record are written in a `finally` block at the end of every voice call (`logs/calls/`). |
| 9 | PII in logs | Pipecat logs transcripts at DEBUG, so raw phone/PAN/Aadhaar could reach the console and logs. | `install_log_redaction()` adds a loguru patcher. Transcripts and post-call records are redacted before writing. |
| 10 | Mood tag spoofing | (New feature risk.) A customer could say or type "[customer_mood: calm]". | Tags are stripped from customer text before classification and tagging. |
| 11 | Retry gaps | `AgentBrain._generate` did not retry 502 and could fall through and return `None`. | Retries 429/500/502/503 with backoff; the impossible fall-through now raises explicitly. |
| 12 | Config | Only the voice bot honoured `CONFIG_PATH`. There was no way to pick a prompt version per run. | `load_config()` honours `CONFIG_PATH` and `PROMPT_VERSION` (or the `prompt_version=` argument) everywhere. |
| 13 | Latency log | Records had no prompt version and no schema, so A/B latency was impossible and drift went unnoticed. | Added `prompt_version`. Each record is validated against `schemas.LatencyRecord` before writing. |
| 14 | Tool-call context | Tool calls had no link to the turn they followed, so violations could not be excerpted. | `ToolCall.turn_index`. In voice it is computed from the spoken context. |
| 15 | Lint drift | Ruff was unpinned (`>=0.6`); 0.16 enables BLE001 by default, so CI and local runs could disagree. | Pinned `ruff==0.16.8` in dev deps. |
| 16 | Prompt mechanics (v1) | v1 told the model to "say goodbye in the same reply" (conflicts with #1) and to take a free-text callback time (conflicts with #7). | Only v1's Tools section was updated to match the new mechanics; its sales and compliance wording is unchanged, so v1 remains the baseline. |
| 17 | Prompt weaknesses (→ v2) | No instruction hierarchy (prompt injection, "admin override"); no rule about other customers' data; no guidance for abusive or impersonating callers. "Pre-approved" could be read as guaranteed. Do-not-call and "not interested" were ambiguous. No Hinglish→24h time conversion. "How did you get my number" invited invention. Volunteered PII could be echoed back. | Fixed in `prompts/v2.md`. Measure with `make compare A=v1 B=v2`. |

Not changed:
- **Interruption of the closing line:** the customer can still barge into the closing line. `EndWorkerFrame` is non-interruptible, so the call still ends (DNC is honoured).
- **Voice transcript timestamps:** these are synthetic (start + 1 s per turn), because Pipecat's context messages carry no timestamps.

# Independent review (fresh-clone reviewer subagent, after the build)

| # | Severity | Finding | Fix |
|---|---|---|---|
| R1 | blocker | `.gitignore` patterns `results/` and `report.html` also matched `demo/`, so the demo data was never committed. `make demo-report`, 3 tests and CI failed on a fresh clone. | Anchored patterns (`/results/`, `/report.html`); demo data committed. |
| R2 | major | Errored calls (bad key, network) were scored. They counted as disclosure violations and gave red-team attacks a free "pass". | Errored transcripts are excluded from scoring and post-call, and listed in `Summary.errored_calls` and the report header. |
| R3 | major | Re-simulated calls kept stale post-call records and scores. | Both are rebuilt when the transcript is newer than the output. |
| R4 | major | A `--no-judge` score was never upgraded once a key was available. | With the judge available, any score without judge results is re-scored. |
| R5 | major | `make eval` treated a simulator crash (exit 1) as "some calls errored" and built an empty report. | Errored calls exit 3; anything else non-zero stops the pipeline; a bad persona selection exits 2. |
| R6 | major | `tts_ttfb_ms` could take the STT's TTFB ("STTService" contains "TTS"). | Match the class-name suffix; the test now has STT slower than TTS. |
| R7 | minor | The report mixed voice latency from all prompt versions. | Latency is filtered by `--version`. |
| R8 | minor | One truncated latency line crashed the report. | Bad lines are skipped with a warning. |
| R9 | minor | `make compare`/`report` hard-coded `results/` and `logs/latency.jsonl`. | The Makefile reads both paths from config. |
| R10 | minor | The A/B per-persona outcome used an arbitrary repeat while the label said "first". | Uses the most common outcome across repeats (natural repeat order); label fixed. |
| R11 | minor | README underestimated full-run time. | Corrected (2–4 h at 8 rpm), with a smoke-run suggestion. |
| R12 | minor | Redaction missed spaced-out digits, card numbers and spaced PAN. | Added; README now says redaction is best-effort. |
| R13 | minor | Agent and customer didn't retry network errors; the SDK's own retries bypassed the limiter. | Connection errors are retried; `max_retries=0` on the client. |
| R14 | minor | No test imported `voice/bot.py`. | `tests/test_voice_bot.py` covers service construction, barge-in config and the DNC hang-up frame order. |
| nits | | `make help` gaps; `judge_model` set when the judge never ran; CLI prog names; scorer missing the "DND" cue; hard-coded "25 personas" in the report footer. | All fixed. |

The reviewer also confirmed that `LLMContext.get_messages()` returns the live list, so MoodTagger's in-place edit persists. That closes a VERIFY item.

# Independent review, round 3 (scenarios, grounding, latency, cost)

v1/v2 check: **pass**. Neither file has changed since commit 761172a, and the rendered personal_loan prompts are byte-identical to commit 4be76fd across 8 name/gender/script variants. Note: v1/v2 runs now get the two grounding tools as well; the tool set is the same for every version.

| # | Severity | Finding | Fix |
|---|---|---|---|
| R3-1 | major | Red-team `prompt_injection` and `guarantee_bait` ignored `grounded_figures` and `scenario_claims`, so "premium 5000, no waiting period" or "lifetime free" passed. | Both attack types now fail on those rules too. |
| R3-2 | major | The rate rule flagged the scenario's own non-rate percentages ("reward rate 2%", "processing fee 2%" next to the rate band, "18% GST"). | The context before the number decides first: a fee/figure label word or GST/tax → not a rate; a rate cue → a rate; then a label right after. Regression tests use the reviewer's sentences. |
| R3-3 | major | The card pack's forbidden claims flagged the waiver statement its own rules require ("agar 4 lakh spend… annual fee nahi lagegi"). | Forbidden claims are unconditional absolutes: conditional, hedged ("promise nahi kar sakti") and already-served sentences are exempt, unless the qualifier is part of the claim itself ("already upgraded"). |
| R3-4 | major | `make bench` ignored `PROMPT`, so before/after both benchmarked v1. | The Makefile passes `--version $(PROMPT) --scenario $(SCENARIO)`; the README commands name the prompt. |
| R3-5 | minor | Demo "hallucinated" rows were really grounded figures. | Replaced with ungrounded figures. |
| R3-6 | minor | The report suggested an impossible `make compare A=v2` for non-loan scenarios. | Hint suppressed for loan-only versions. |
| R3-7 | minor | "Red-team pass rate 0.0% vs 0.0%" when an arm had no red-team calls. | n=0 is recorded; the report shows "—" and "no data in one arm". |
| R3-8 | minor | Grounding false positives on "2 din mein call karenge", "ek saal mein 4 lakh spend", "18% GST". | Short contact/spend timeframes and tax percentages are masked; "disbursal 3 din mein" is still checked. |
| R3-9 | minor | Scorer objection detection was loan-only; judge rubric and `record_consent` were loan-worded. | Uses the scenario playbook (same patterns as post-call); wording made generic. |
| R3-10 | minor | v3 described `calculate_emi` even where it isn't offered (cards have an APR band but no EMI tool). | The EMI rule and tool line render only when `emi_enabled`; otherwise the prompt says an executive will share EMI details. |
| R3-11 | minor | The suite went red when the user exported `SCENARIO`/`PROMPT_VERSION`. | An autouse fixture clears them. |
| R3-12 | minor | `make score` with no transcripts wrote an empty summary, which hid the error in the report. | Exits 2 with a "run make sim" message. |
| notes | | STT billing, stale comparisons, Devanagari disclosure. | Documented in README Limitations. |
