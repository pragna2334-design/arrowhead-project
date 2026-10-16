"""DeepSeek thinking (reasoning) flag: validation and the startup log line (pure, no Pipecat).

Shared by the voice bot and the latency bench so both run the model the same way.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

THINKING_VALUES = ("enabled", "disabled")


def validate_thinking(value: object) -> str:
    """`brain.thinking` must be exactly "enabled" or "disabled" (DeepSeek's own values)."""
    if value not in THINKING_VALUES:
        raise ValueError(
            f"brain.thinking must be one of {THINKING_VALUES}, got {value!r} "
            "(use 'disabled' for voice: reasoning adds ~0.7 s before the first word)"
        )
    return str(value)


def thinking_log_line(model: str, thinking: str) -> str:
    state = "reasoning off" if thinking == "disabled" else "REASONING ON, slow first token"
    return f"LLM {model}: thinking={thinking} ({state})"


def log_thinking(model: str, thinking: str, log: Any = logger) -> str:
    """Log one clear startup line; warn loudly when reasoning is on. Returns the line."""
    line = thinking_log_line(model, validate_thinking(thinking))
    if thinking == "enabled":
        log.warning(f"{line}. Set brain.thinking: disabled in config.yaml for voice latency.")
    else:
        log.info(line)
    return line
