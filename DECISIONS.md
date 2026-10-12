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
- Mood is tagged on the context message (MoodTagger processor), not on the transcription frame — the browser transcript stays clean while the LLM sees the same tag as in text mode. `VERIFY` that the in-place context edit persists.
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
- Errors still write the transcript with `stop_reason: error`, and `make eval` continues past them — no lost evidence; re-running resumes the failed calls.
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
- `demo/` is synthetic data, deterministic and clearly labelled (`demo-` ids, `-demo` models, `[demo]` text, a banner) — so anyone can see the report without keys, and it can't be mistaken for real results.

## Not verified (needs live keys or a browser)
DeepSeek model ids and JSON mode alongside the thinking setting; Deepgram `multi` on Hinglish; Sarvam pronunciation; the goodbye audio playing fully before hang-up; MoodTagger's context edit persisting; real latency numbers; how the simulated customers behave; the report in a real browser.
