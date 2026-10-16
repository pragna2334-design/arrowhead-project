import re

import pytest

from brain.prompt import available_versions, build_system_prompt


@pytest.mark.parametrize("version", available_versions())
def test_every_prompt_renders(cfg, version):
    text = build_system_prompt(cfg.brain, "Rahul Verma", version)
    assert not re.search(r"\{[a-z_]+\}", text), "unfilled placeholder"
    assert cfg.brain.lender_name in text and "Rahul Verma" in text
    assert "mark_do_not_call" in text and "customer_mood" in text
    if version in ("v1", "v2"):  # loan-only prompts state the band directly
        assert f"{cfg.brain.rate_min_pct}%" in text and f"{cfg.brain.rate_max_pct}%" in text


def test_v1_exists_and_config_selects_it(cfg):
    assert "v1" in available_versions()
    assert build_system_prompt(cfg.brain, "X") == build_system_prompt(
        cfg.brain, "X", cfg.brain.prompt_version
    )


def test_unknown_version_errors(cfg):
    with pytest.raises(FileNotFoundError):
        build_system_prompt(cfg.brain, "X", "v999")


def test_judge_must_differ(tmp_path):
    import yaml

    from brain.config import DEFAULT_CONFIG_PATH, load_config

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["brain"]["judge_model"] = raw["brain"]["agent_model"]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(p)


def test_prompt_version_override(cfg):
    from brain.config import load_config

    assert load_config(prompt_version="v1").brain.prompt_version == "v1"
