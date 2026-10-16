"""Text-mode simulator: an LLM plays each persona in sim/personas.yaml against AgentBrain.

    python -m sim.simulate [--version v1] [--n 25] [--personas id1,id2] [--repeats 1]
                           [--fresh] [--results results]

Writes one Transcript per call to results/<version>/transcripts/<call_id>.json and keeps
resume state in results/<version>/sim_state.json. Never scores and never builds post-call
records: those are separate steps, so re-scoring never re-simulates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from brain.agent import AgentBrain, make_client
from brain.config import AppConfig, load_config
from brain.prompt import available_versions
from brain.state import CallState
from brain.transcript import to_transcript
from schemas import ResultsLayout, Transcript
from sim.customer import HANGUP, CustomerSim
from sim.ratelimit import RateLimiter

PERSONAS_PATH = Path(__file__).resolve().parent / "personas.yaml"
PERSONA_KEYS = (
    "id",
    "customer_name",
    "language",
    "description",
    "behaviour",
    "red_team",
    "attack_type",
)
LANGUAGES = ("hinglish", "hindi", "english")
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


# ---------------------------------------------------------------- personas


def load_personas(path: str | Path = PERSONAS_PATH) -> list[dict[str, Any]]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    personas = data.get("personas") or []
    seen: set[str] = set()
    for p in personas:
        missing = [k for k in PERSONA_KEYS if k not in p]
        if missing:
            raise ValueError(f"persona {p.get('id')!r} missing keys {missing}")
        if p["id"] in seen:
            raise ValueError(f"duplicate persona id {p['id']!r}")
        if p["language"] not in LANGUAGES:
            raise ValueError(f"persona {p['id']!r}: language must be one of {LANGUAGES}")
        seen.add(p["id"])
    return personas


def select_personas(
    personas: list[dict[str, Any]], n: int | None = None, ids: list[str] | None = None
) -> list[dict[str, Any]]:
    if ids:
        by_id = {p["id"]: p for p in personas}
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            raise ValueError(f"unknown persona ids: {unknown}")
        return [by_id[i] for i in ids]
    return personas[:n] if n is not None else list(personas)


def make_call_id(version: str, persona_id: str, repeat: int) -> str:
    return _SAFE.sub("_", f"{version}-{persona_id}-{repeat}")


# ---------------------------------------------------------------- resume state


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


class SimState:
    """results/<version>/sim_state.json: per-call status and timestamps."""

    def __init__(self, path: Path, version: str):
        self.path = path
        self.data: dict[str, Any] = {"version": version, "calls": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
                self.data.setdefault("calls", {})
            except (OSError, json.JSONDecodeError):
                pass

    def mark(self, call_id: str, status: str, **fields: Any) -> None:
        entry = self.data["calls"].setdefault(call_id, {})
        entry.update(status=status, **fields)
        if status == "started":
            for k in ("finished_at", "stop_reason", "error"):
                entry.pop(k, None)
        calls = self.data["calls"]
        for s in ("started", "finished", "errored"):
            self.data[s] = sorted(c for c, e in calls.items() if e.get("status") == s)
        self.data["updated_at"] = time.time()
        _write_json(self.path, self.data)


def is_done(path: Path) -> bool:
    """A call is done if its transcript exists, validates and did not stop on an error."""
    try:
        return Transcript.model_validate_json(path.read_text()).stop_reason != "error"
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------- one call


async def run_call(
    persona: dict[str, Any],
    call_id: str,
    *,
    app: AppConfig,
    agent_client: Any,
    customer_client: Any,
    limiter: RateLimiter | None = None,
) -> Transcript:
    """Simulate one call and return its (redacted) Transcript. Never raises on LLM errors."""
    cfg, sim = app.brain, app.sim
    max_turns = int(sim.get("max_turns", 16))
    started_at = time.time()
    stop_reason, error = "max_turns", None
    brain: AgentBrain | None = None
    try:
        brain = AgentBrain(
            cfg, persona["customer_name"], call_id, client=agent_client, before_request=limiter
        )
        customer = CustomerSim(
            persona,
            model=sim.get("customer_model", cfg.agent_model),
            temperature=float(sim.get("customer_temperature", 0.8)),
            client=customer_client,
            lender_name=cfg.lender_name,
            before_request=limiter,
            usage=brain.state.usage.customer,
        )
        result = await brain.opening()
        if result.ended:
            stop_reason = "agent_ended"
        else:
            for _ in range(max_turns):
                said = await customer.reply(brain.state.turns)
                if said == HANGUP:
                    stop_reason = "customer_hung_up"
                    break
                result = await brain.respond(said)
                if result.ended:
                    stop_reason = "agent_ended"
                    break
    except Exception as e:  # noqa: BLE001 - any failure is recorded on the transcript
        stop_reason, error = "error", f"{type(e).__name__}: {e}"
    state = brain.state if brain else CallState(call_id, persona["customer_name"])
    return to_transcript(
        state,
        mode="text",
        prompt_version=cfg.prompt_version,
        agent_model=cfg.agent_model,
        started_at=started_at,
        stop_reason=stop_reason,
        persona_id=persona["id"],
        error=error,
        scenario=cfg.scenario.id,
    )


# ---------------------------------------------------------------- batch


async def run_sim(
    app: AppConfig,
    *,
    personas: list[dict[str, Any]],
    repeats: int = 1,
    fresh: bool = False,
    results_dir: str | Path | None = None,
    agent_client: Any = None,
    customer_client: Any = None,
    limiter: RateLimiter | None = None,
    log=print,
) -> dict[str, str]:
    """Run every (persona, repeat) sequentially. Returns {call_id: stop_reason | "skipped"}."""
    version = app.brain.prompt_version
    layout = ResultsLayout(root=Path(results_dir or app.results_dir), version=version)
    layout.transcripts.mkdir(parents=True, exist_ok=True)
    state = SimState(layout.sim_state, version)
    if limiter is None:
        limiter = RateLimiter(float(app.sim.get("rpm", 8)))
    if agent_client is None or customer_client is None:
        shared = make_client()
        agent_client = agent_client or shared
        customer_client = customer_client or shared

    jobs = [(p, r) for p in personas for r in range(1, repeats + 1)]
    outcome: dict[str, str] = {}
    for i, (persona, repeat) in enumerate(jobs, 1):
        call_id = make_call_id(version, persona["id"], repeat)
        path = layout.transcripts / f"{call_id}.json"
        prefix = f"[{i}/{len(jobs)}] {call_id}"
        if not fresh and is_done(path):
            outcome[call_id] = "skipped"
            log(f"{prefix}: skipped (already done)")
            continue
        state.mark(call_id, "started", persona_id=persona["id"], started_at=time.time())
        t0 = time.time()
        tr = await run_call(
            persona,
            call_id,
            app=app,
            agent_client=agent_client,
            customer_client=customer_client,
            limiter=limiter,
        )
        _write_json(path, tr.model_dump(mode="json"))
        state.mark(
            call_id,
            "errored" if tr.stop_reason == "error" else "finished",
            finished_at=time.time(),
            stop_reason=tr.stop_reason,
            error=tr.error,
        )
        outcome[call_id] = tr.stop_reason
        fs = tr.final_state
        log(
            f"{prefix}: {tr.stop_reason} turns={len(tr.turns)} end={fs.end_reason} "
            f"consent={fs.consent} callback={fs.callback_time} dnc={fs.dnc} "
            f"{time.time() - t0:.0f}s" + (f" error={tr.error}" if tr.error else "")
        )
    return outcome


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.simulate", description=__doc__.split("\n")[0])
    ap.add_argument("--version", help="prompt version (default: config brain.prompt_version)")
    ap.add_argument("--n", type=int, help="first N personas (default: config sim.persona_count)")
    ap.add_argument("--personas", help="comma-separated persona ids (overrides --n)")
    ap.add_argument("--repeats", type=int, default=1, help="runs per persona (default 1)")
    ap.add_argument("--fresh", action="store_true", help="re-run calls that are already done")
    ap.add_argument("--results", help="results root dir (default: config paths.results_dir)")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv  # project dependency; same as voice/bot.py

        load_dotenv()
    except ImportError:
        pass
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY is not set (put it in .env)", file=sys.stderr)
        return 2

    app = load_config(prompt_version=args.version)
    if app.brain.prompt_version not in available_versions():
        print(f"unknown prompt version {app.brain.prompt_version!r}", file=sys.stderr)
        return 2
    ids = [s.strip() for s in args.personas.split(",") if s.strip()] if args.personas else None
    n = args.n if args.n is not None else app.sim.get("persona_count")
    try:
        personas = select_personas(load_personas(), n, ids)
    except (ValueError, KeyError, OSError) as e:
        print(f"bad persona selection: {e}", file=sys.stderr)
        return 2
    print(
        f"sim {app.brain.prompt_version}: {len(personas)} personas x {args.repeats} "
        f"(rpm={app.sim.get('rpm')}, max_turns={app.sim.get('max_turns')})"
    )
    outcome = asyncio.run(
        run_sim(
            app,
            personas=personas,
            repeats=max(1, args.repeats),
            fresh=args.fresh,
            results_dir=args.results,
        )
    )
    errors = sum(1 for v in outcome.values() if v == "error")
    print(f"done: {len(outcome)} calls, {errors} errors")
    return EXIT_CALLS_ERRORED if errors else 0


# Exit codes: 0 ok, 2 setup error (stop the pipeline), 3 some calls errored (others are fine;
# `make eval` continues and the scorer excludes errored calls). Anything else is a crash.
EXIT_CALLS_ERRORED = 3


if __name__ == "__main__":
    raise SystemExit(main())
