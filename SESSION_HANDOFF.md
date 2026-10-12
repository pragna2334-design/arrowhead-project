# Session handoff

Last updated: 2026-10-12. Read this first in a new session, then `README.md`, `DECISIONS.md` and `AUDIT.md`.

## Where things stand

- **Repo:** `pragna2334-design/arrowhead-project`. All work is on `main`, and everything is pushed.
  - GitHub's default branch should be set to `main` in repo settings.
  - The old `claude/sharp-cannon-j8ivct` branch only has the first README.
- **Built:** everything in the spec (phases 1–3 and the extended brief):
  - Voice bot: Pipecat + SmallWebRTC, Deepgram STT, Sarvam `bulbul:v3` voice `priya`, DeepSeek LLM.
  - Shared brain, with versioned prompts `v1` and `v2`.
  - Simulator: 25 personas, 5 of them red-team.
  - Scorer: 6 rules plus an LLM judge.
  - Post-call CRM records and PII redaction.
  - Static report with an A/B view; `make eval`, `make compare` and `make demo-report`.
  - CI.
  - Independent review: 14 findings, all fixed.
- **Tests:** 279 offline tests, all passing. Lint is clean.
- **LLM:** DeepSeek only (the user has no Gemini key). Agent model `deepseek-flash`; judge model `deepseek-pro`, which is a **guess**. If the judge errors, change `brain.judge_model` in `config.yaml`.
- **User's machine:** WSL Ubuntu, repo at `~/arrowhead-project`. `make install` and `make demo-report` work. Keys are in `.env`: DEEPGRAM, SARVAM, DEEPSEEK.

## Verified by the user locally

- `make demo-report` builds.
- **Voice calls work end to end:**
  - Hinglish conversation, female voice.
  - Hang-up through a tool call, then the closing line.
  - `logs/latency.jsonl` and `logs/calls/<id>.{transcript,postcall}.json` are written.

## Latest latency (voice-9465ee35, prompt v1, 4 reply turns)

| | p50 | range |
|---|---|---|
| `e2e_ms` | ~1,550 | 1,383–1,834 |
| `pipeline_ms` | ~1,175 | 1,031–1,347 |

Breakdown of a typical turn:

| Stage | ms |
|---|---|
| Endpointing | 200 |
| STT | ~250 |
| **LLM until first text** | **~800–1,070 (~60% of the turn)** |
| Sentence aggregation | ~45–100 |
| Sarvam TTS | ~175 |

- **The LLM gap:** `llm_ttfb_ms` is only ~180 ms, but the first *speakable text* takes ~700 ms more. DeepSeek may still be reasoning despite `thinking: disabled`, or that may simply be its real time to first token. This is not yet diagnosed.
- **The "greeting" with e2e 17,509 ms (turn 2):** a Pipecat observer quirk. It anchored the first reply to client connect; the line is labelled `greeting` and excluded from the percentiles. The first real reply's latency is lost each call.
- **The 500 ms target** is not reachable end to end: endpointing plus STT is already ~450 ms. A realistic goal is ~1.0–1.2 s. Consider stating the target against `pipeline_ms` in the README.

## Pending: waiting on the user

1. **Run the DeepSeek reasoning test and paste the output.**
   ```bash
   .venv/bin/python - <<'E'
   import os, time
   from dotenv import load_dotenv; load_dotenv()
   from openai import OpenAI
   c = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
   t = time.time(); first = None; reasoning = 0
   for ch in c.chat.completions.create(model="deepseek-flash", stream=True,
           messages=[{"role": "user", "content": "Ek line mein namaste bolo"}],
           extra_body={"thinking": {"type": "disabled"}}):
       d = ch.choices[0].delta if ch.choices else None
       if d and getattr(d, "reasoning_content", None): reasoning += 1
       if d and d.content and first is None: first = time.time() - t
   print(f"first content token: {first*1000:.0f} ms, reasoning chunks: {reasoning}")
   E
   ```
   - Reasoning chunks above 0: thinking isn't really off. Fixing that would save ~700 ms.
   - 0 chunks with ~800 ms or more to the first token: that is DeepSeek's real speed.
2. **Results of the four voice checks.** The transcript is in `logs/calls/voice-9465ee35.transcript.json`.
   - "Rate bahut zyada hai": acknowledges, gives the range, promises no lower rate.
   - "Kal raat 9 baje call karna": refuses 9 pm and offers a time between 10:00 and 19:00.
   - "haan" / "hmm" while the bot talks: must **not** interrupt; a full sentence should.
   - "Mujhe dobara call mat karna": DNC confirmation, then hang-up.
3. **Which latency changes to make.** Recommended: options 1 and 2 below in any case, and option 3 if the test shows reasoning chunks.

## Proposed next changes (not started)

1. **`text_aggregation_mode: token` for Sarvam TTS.**
   - Make it a config switch (`tts.sarvam.text_aggregation_mode`).
   - `SarvamTTSService` accepts `text_aggregation_mode=TextAggregationMode.TOKEN` (`pipecat.services.tts_service`). Sarvam buffers about 50 characters itself.
   - Saves about 50–150 ms.
2. **Prompt `v3` = v2 plus "start every reply with a 2–3 word acknowledgement"** ("Ji, samajh gayi."). The first sentence ends sooner, so TTS starts sooner. Measure with `make compare A=v2 B=v3` and voice latency per version; the report filters latency by `--version`.
3. **If DeepSeek is reasoning anyway:** look at a different model id or a thinking parameter. The service is `pipecat/services/deepseek/llm.py`, which sends `extra_body={"thinking": ...}`.
4. **Possibly fix the turn-2 "greeting" anchoring:** investigate `UserBotLatencyObserver`'s `MeasuredFrom` logic with the mute strategies.

## Still to do after that

```bash
make eval PROMPT=v1 ARGS="--n 3"   # smoke test first
make eval PROMPT=v1                # 25 personas; 2-4 h at 8 rpm (raise sim.rpm / score.rpm if quota allows)
make eval PROMPT=v2
make compare A=v1 B=v2             # open report.html  (WSL: explorer.exe report.html)
```

Then:
- Fill in the **Results** table and the `_X_/_Y_/_A_/_B_/_N_` placeholders in README's resume summary.
- Clear the remaining `VERIFY` items in DECISIONS.md:
  - the model ids
  - Deepgram `multi` on Hinglish
  - Sarvam pronunciation
  - whether the goodbye audio plays fully before hang-up and the closing line reaches the voice transcript
  - how the simulated customers behave
  - how the report looks in a browser

## Useful facts for the next session

- **Pipecat is pinned at 1.12.0.** Check APIs against `.venv/lib/python3.11/site-packages/pipecat`; docs.pipecat.ai was unreachable from the cloud sandbox.
- **Where the code lives:**
  - Shared brain: `brain/agent.py` (text loop), `brain/tools.py` (tools, callback-hours check, closing lines), `brain/sentiment.py` (mood tags), `brain/prompt.py` plus `prompts/*.md`.
  - Voice: `voice/bot.py` (pipeline, barge-in and mute strategies, tool handlers, post-call), `voice/latency.py`, `voice/mood.py`.
  - Data contracts: `schemas/` (frozen pydantic). Results go to `results/<version>/{transcripts,postcall,scores,summary.json}`; comparisons to `results/compare_<a>_<b>.json`.
- **Exit codes:** the simulator exits 3 when only some calls errored (the pipeline continues, and errored calls are excluded from scoring), and 2 on a setup error.
- **Commits** end with the session's Co-Authored-By and Claude-Session trailers. Push to `main`.
