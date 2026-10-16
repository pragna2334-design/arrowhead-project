"""System prompt builder. Prompt text lives in prompts/<version>.md; config picks the version."""

from __future__ import annotations

from pathlib import Path

from brain.config import BrainConfig

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Sent as the first user turn in both modes so the agent speaks first (outbound call).
OPENING_CUE = "(The customer has just picked up the phone. Begin the call.)"


def available_versions() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("v*.md"))


# v1/v2 were written for the personal-loan pitch and hard-code its terms; they stay frozen so
# the v1-vs-v2 comparison remains clean. v3+ are scenario-generic templates.
LOAN_ONLY_VERSIONS = {"v1", "v2"}


def _ack_examples(female: bool) -> str:
    understood = "samajh gayi" if female else "samajh gaya"
    return f'"Haan ji,", "Ji, {understood}.", "Achha,", "Bilkul,", "Theek hai,"'


def _objection_playbook(cfg: BrainConfig) -> str:
    plays = cfg.scenario.objections
    if not plays:
        return "- Handle objections honestly and briefly; never pressure."
    return "\n".join(f"- {o.id.replace('_', ' ')}: {o.guidance}" for o in plays)


def _scenario_rules(cfg: BrainConfig) -> str:
    rules = list(cfg.scenario.compliance.extra_rules)
    return "\n".join(f"- {r}" for r in rules) if rules else "- No additional product rules."


def build_system_prompt(cfg: BrainConfig, customer_name: str, version: str | None = None) -> str:
    version = version or cfg.prompt_version
    path = PROMPTS_DIR / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"prompt version {version!r} not found; have {available_versions()}"
        )
    if version in LOAN_ONLY_VERSIONS and cfg.scenario.id != "personal_loan":
        raise ValueError(
            f"prompt {version} only supports scenario 'personal_loan' (got {cfg.scenario.id!r}); "
            "use v3 or later for other scenarios"
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
    common = {
        "agent_name": cfg.agent_name,
        "customer_name": customer_name,
        "product": cfg.product,
        "script_rule": script_rule,
        "gender_rule": gender_rule,
        "callback_start": cfg.callback_hours[0],
        "callback_end": cfg.callback_hours[1],
    }
    if version in LOAN_ONLY_VERSIONS:
        # Exactly the Phase-2 rendering (pinned by tests/fixtures/prompts/*).
        return path.read_text().format(
            **common,
            lender_name=cfg.lender_name,
            rate_min=cfg.rate_min_pct,
            rate_max=cfg.rate_max_pct,
            max_amount=f"{cfg.max_amount_inr:,}",
            tenure_min=cfg.tenure_months[0],
            tenure_max=cfg.tenure_months[1],
            opening_example=(
                f"Namaste, main {cfg.agent_name} {verb}, {cfg.lender_name} ki taraf se AI "
                f"assistant. Kya meri baat {customer_name} ji se ho rahi hai?"
            ),
        )
    sc = cfg.scenario
    return path.read_text().format(
        **common,
        company=sc.company,
        company_kind=sc.company_kind,
        product_pitch=sc.product.pitch,
        objection_playbook=_objection_playbook(cfg),
        scenario_rules=_scenario_rules(cfg),
        ack_examples=_ack_examples(female),
        opening_example=sc.opening.format(
            agent_name=cfg.agent_name, verb=verb, company=sc.company, customer_name=customer_name
        ),
    )
