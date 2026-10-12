"""Persona lookup for the scorer (description + red-team flags), tolerant of a missing file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from score.rules import ATTACK_TYPES, red_team_attack_from_id

PERSONAS_PATH = Path(__file__).resolve().parent.parent / "sim" / "personas.yaml"


@dataclass(frozen=True)
class PersonaInfo:
    id: str | None
    description: str | None = None
    red_team: bool = False
    attack_type: str | None = None


def load_persona_map(path: str | Path = PERSONAS_PATH) -> dict[str, dict[str, Any]]:
    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return {
        str(p["id"]): p for p in (data.get("personas") or []) if isinstance(p, dict) and p.get("id")
    }


def persona_info(persona_id: str | None, personas: dict[str, dict[str, Any]]) -> PersonaInfo:
    p = personas.get(persona_id or "")
    fallback_rt, fallback_attack = red_team_attack_from_id(persona_id)
    if p is None:
        return PersonaInfo(persona_id, None, fallback_rt, fallback_attack)
    desc = " ".join(str(p.get(k) or "").strip() for k in ("description", "behaviour")).strip()
    red = bool(p.get("red_team")) or fallback_rt
    attack = p.get("attack_type") if p.get("attack_type") in ATTACK_TYPES else fallback_attack
    return PersonaInfo(persona_id, desc or None, red, attack if red else None)
