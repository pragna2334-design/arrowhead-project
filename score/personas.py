"""Persona lookup for the scorer (description + red-team flags), tolerant of a missing file.

Personas live in each scenario pack (scenarios/<scenario>/personas.yaml). Use
`persona_info_for(t)` to resolve a transcript's persona against its own scenario; unknown
personas (or a missing file) fall back to rt_<attack_type> id inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from brain.scenario import DEFAULT_SCENARIO, personas_path
from schemas import Transcript
from score.rules import ATTACK_TYPES, red_team_attack_from_id

PERSONAS_PATH = personas_path(DEFAULT_SCENARIO)
_SCENARIO_ID = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class PersonaInfo:
    id: str | None
    description: str | None = None
    red_team: bool = False
    attack_type: str | None = None


def load_persona_map(
    path: str | Path | None = None, *, scenario: str | None = None
) -> dict[str, dict[str, Any]]:
    """{persona_id: persona} from `path`, else from the scenario's pack (default
    personal_loan). Missing or unreadable file -> {} (callers fall back to id inference)."""
    if path is None:
        scenario = scenario or DEFAULT_SCENARIO
        if not _SCENARIO_ID.fullmatch(scenario):  # ids come from transcripts: no path tricks
            return {}
        path = personas_path(scenario)
    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
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


def persona_info_for(
    t: Transcript, cache: dict[str, dict[str, dict[str, Any]]] | None = None
) -> PersonaInfo:
    """PersonaInfo for a transcript, looked up in the transcript's own scenario pack.

    Pass the same `cache` dict across calls to read each scenario's personas.yaml once.
    """
    scenario = t.scenario or DEFAULT_SCENARIO
    if cache is None:
        personas = load_persona_map(scenario=scenario)
    else:
        if scenario not in cache:
            cache[scenario] = load_persona_map(scenario=scenario)
        personas = cache[scenario]
    return persona_info(t.persona_id, personas)
