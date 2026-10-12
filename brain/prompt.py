"""System prompt builder. Prompt text lives in prompts/<version>.md; config picks the version."""

from __future__ import annotations

from pathlib import Path

from brain.config import BrainConfig

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Sent as the first user turn in both modes so the agent speaks first (outbound call).
OPENING_CUE = "(The customer has just picked up the phone. Begin the call.)"


def available_versions() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("v*.md"))


def build_system_prompt(cfg: BrainConfig, customer_name: str, version: str | None = None) -> str:
    version = version or cfg.prompt_version
    path = PROMPTS_DIR / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"prompt version {version!r} not found; have {available_versions()}"
        )
    female = cfg.agent_gender == "female"
    verb = "bol rahi hoon" if female else "bol raha hoon"
    script_rule = (
        f"Write Hindi words in Roman script (Hinglish), e.g. 'Namaste, main {cfg.agent_name} {verb}'."
        if cfg.script == "roman"
        else "Write Hindi words in Devanagari and English words in Latin script."
    )
    gender_rule = (
        "You are female: always use feminine Hindi verb forms about yourself "
        "(rahi hoon, sakti hoon, karungi)."
        if female
        else "You are male: always use masculine Hindi verb forms about yourself "
        "(raha hoon, sakta hoon, karunga)."
    )
    return path.read_text().format(
        agent_name=cfg.agent_name,
        lender_name=cfg.lender_name,
        customer_name=customer_name,
        product=cfg.product,
        rate_min=cfg.rate_min_pct,
        rate_max=cfg.rate_max_pct,
        max_amount=f"{cfg.max_amount_inr:,}",
        tenure_min=cfg.tenure_months[0],
        tenure_max=cfg.tenure_months[1],
        script_rule=script_rule,
        gender_rule=gender_rule,
        callback_start=cfg.callback_hours[0],
        callback_end=cfg.callback_hours[1],
        opening_example=(
            f"Namaste, main {cfg.agent_name} {verb}, {cfg.lender_name} ki taraf se AI assistant. "
            f"Kya meri baat {customer_name} ji se ho rahi hai?"
        ),
    )
