"""Cost contracts: token usage and the per-call cost record."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Token usage for one LLM role, summed over a call."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0  # prompt tokens served from the provider cache, if reported

    def add(self, prompt: int, completion: int, cached: int = 0) -> None:
        self.requests += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cached_prompt_tokens += cached


class CallUsage(BaseModel):
    """Usage recorded on a Transcript (agent + simulated customer)."""

    agent: Usage = Field(default_factory=Usage)
    customer: Usage = Field(default_factory=Usage)  # simulator only; zero for voice
    agent_chars: int = 0  # characters the agent spoke (drives TTS cost)
    customer_words: int = 0  # words the customer spoke (drives STT cost estimate)


class CostRecord(BaseModel):
    """Written to results/<scenario>/<version>/costs/<call_id>.json by the scorer.

    Production cost excludes the simulated customer (it does not exist in production);
    eval cost includes customer + judge (what running this evaluation costs you).
    """

    schema_version: int = 1
    call_id: str
    scenario: str
    prompt_version: str
    mode: str  # text | voice
    agent_usage: Usage
    customer_usage: Usage
    judge_usage: Usage
    stt_seconds: float  # measured (voice) or estimated from customer words (text)
    tts_seconds: float  # measured (voice) or estimated from agent characters (text)
    tts_chars: int
    audio_estimated: bool  # True when seconds were estimated from text
    breakdown_usd: dict[str, float]  # llm_agent, llm_customer, llm_judge, stt, tts
    production_usd: float  # llm_agent + stt + tts
    eval_usd: float  # production + llm_customer + llm_judge
    price_table: str  # config price-table id, e.g. "2026-10"
