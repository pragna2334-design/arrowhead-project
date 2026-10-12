# Decisions (one line each)

- Pinned `pipecat-ai==1.12.0` (latest on PyPI); all imports checked against the installed 1.12.0 source and the v1.12.0 `voice-sarvam.py` example.
- docs.pipecat.ai was unreachable, so source on GitHub/PyPI was used instead of rendered docs.
- Python 3.11 venv via `uv` (spec says 3.11); sandbox default is 3.13.
- LLM is DeepSeek everywhere (user has a DeepSeek key, no Gemini key); Gemini and `google-genai` were removed. Agent `deepseek-flash` (Pipecat 1.12.0's current name for V4.1 Flash); judge `deepseek-pro` (a guess; differs from agent as required). `# VERIFY`: both ids on your key; DeepSeek docs were unreachable.
- DeepSeek thinking is `disabled` in voice and text mode (`brain.thinking`): V4 models otherwise reason before answering, which hurts TTFB. Text mode sends the same `extra_body={"thinking": ...}` field Pipecat sends.
- Text mode uses the `openai` SDK at `https://api.deepseek.com/v1` (Pipecat's default base URL). Pipecat's `DeepSeekLLMAdapter` handles `reasoning_content` in voice; text mode runs thinking-disabled so it is not needed. `# VERIFY` if you enable thinking: text mode would need the same handling.
- Dropped the word "fictional" from the spoken prompt: the agent should not tell callers it is fictional; fiction is in README only.
- Hindi written in Roman script by default (`brain.script`): simplest for the LLM and for Sarvam `hi-IN`. `# VERIFY`: listen test for English-word pronunciation; Devanagari is one config line away.
- Deepgram STT `language: "multi"` passed as a raw string (Pipecat logs it as "not a recognized Language code, passed as-is"). `# VERIFY`: Deepgram accepts it for nova-3 and handles Hinglish; fallback `"hi"` or Sarvam STT.
- Sarvam TTS has no Hinglish code, so `hi-IN` is used and mixed text is sent as-is.
- `mark_do_not_call` also ends the call (compliance: stop immediately); `end_call` after DNC keeps reason `dnc`.
- Terminal tools return with `run_llm=False`; the agent must say goodbye in the same reply before calling them (prompt rule).
- Hang-up uses `EndWorkerFrame` pushed downstream from the LLM so queued speech flushes. `# VERIFY` (live run): the goodbye audio is fully played before disconnect.
- Opening cue (`OPENING_CUE`) is a user-role message in both modes (DeepSeek has no `developer` role).
- Voice mode does not save transcripts; only final call state (consent/callback/dnc) is logged. Transcripts come from the simulator in Phase 2.
- Latency: `pipeline_ms` = sum of observer contributions except endpointing_wait, transcription, first_request, output_transport. `# VERIFY`: contribution keys fire as expected in a live call.
- Customer is always the fictional "Rahul Verma"; configurable in `config.yaml`.
- Agent is female ("Priya", Sarvam voice `priya`, feminine Hindi verb forms via `brain.agent_gender`); first live test showed the default `shubh` voice is male. `# VERIFY`: listen to `priya` vs other female voices.
- Simple retry (429/500/503, 2-32 s backoff) lives in `AgentBrain`; the simulator rate limiter (8 RPM default) is Phase 2; DeepSeek's real limits are unverified.
- ElevenLabs model/voice and Deepgram TTS voice names are unverified placeholders (`# VERIFY` in config.yaml).

## Not verified (needs live keys / a browser)

Deepgram `multi` behaviour, Sarvam Hinglish pronunciation, DeepSeek model ids on your key, DeepSeek tool call + text in one reply streaming before hang-up, WebRTC prebuilt UI end to end, real latency numbers.
