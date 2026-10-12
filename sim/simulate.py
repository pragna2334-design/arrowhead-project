"""TODO (Phase 2): text-mode simulator.

An LLM plays each persona in sim/personas.yaml against `brain.agent.AgentBrain`.
- max 16 turns per call, persona count configurable (--n), default 8 RPM with 429 backoff
- saves one transcript per call as JSON (CallState.to_dict() + persona id) in transcripts/
- never scores; scoring is a separate step so re-scoring never re-simulates
"""
