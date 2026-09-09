"""Pins cellsense.config.load_config's precedence chain (CLI > env >
./.cellsense/config.toml > ~/.cellsense/config.toml > defaults), malformed
TOML / unknown key errors, and redaction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cellsense.config import Config, ModelConfig, _redact, load_config, write_default_config
from cellsense.errors import ConfigError


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test gets its own fake $HOME and no .env in cwd, so real
    developer-machine config/.env files never leak into these tests.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    return fake_home


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _clear_cellsense_env(monkeypatch) -> None:
    for name in [
        "CELLSENSE_MODEL_DEFAULT",
        "CELLSENSE_MODEL_TEMPERATURE",
        "CELLSENSE_AGENT_MAX_TOOL_ROUNDS",
        "CELLSENSE_AGENT_GUARDRAIL",
        "CELLSENSE_PERMISSIONS_MODE",
        "CELLSENSE_PERMISSIONS_ALLOW",
    ]:
        monkeypatch.delenv(name, raising=False)


class TestPrecedence:
    def test_defaults_when_nothing_is_configured(self, cwd, monkeypatch) -> None:
        _clear_cellsense_env(monkeypatch)
        config = load_config({}, cwd=cwd)
        assert config == Config()

    def test_home_toml_overrides_defaults(self, cwd, monkeypatch, _isolated_home) -> None:
        _clear_cellsense_env(monkeypatch)
        home_dir = _isolated_home / ".cellsense"
        home_dir.mkdir()
        (home_dir / "config.toml").write_text('[model]\ndefault = "home-model"\n')
        config = load_config({}, cwd=cwd)
        assert config.model.default == "home-model"

    def test_local_toml_overrides_home_toml(self, cwd, monkeypatch, _isolated_home) -> None:
        _clear_cellsense_env(monkeypatch)
        home_dir = _isolated_home / ".cellsense"
        home_dir.mkdir()
        (home_dir / "config.toml").write_text('[model]\ndefault = "home-model"\n')
        local_dir = cwd / ".cellsense"
        local_dir.mkdir()
        (local_dir / "config.toml").write_text('[model]\ndefault = "local-model"\n')
        config = load_config({}, cwd=cwd)
        assert config.model.default == "local-model"

    def test_env_overrides_local_toml(self, cwd, monkeypatch, _isolated_home) -> None:
        _clear_cellsense_env(monkeypatch)
        local_dir = cwd / ".cellsense"
        local_dir.mkdir()
        (local_dir / "config.toml").write_text('[model]\ndefault = "local-model"\n')
        monkeypatch.setenv("CELLSENSE_MODEL_DEFAULT", "env-model")
        config = load_config({}, cwd=cwd)
        assert config.model.default == "env-model"

    def test_cli_overrides_env(self, cwd, monkeypatch, _isolated_home) -> None:
        _clear_cellsense_env(monkeypatch)
        monkeypatch.setenv("CELLSENSE_MODEL_DEFAULT", "env-model")
        config = load_config({"model": {"default": "cli-model"}}, cwd=cwd)
        assert config.model.default == "cli-model"

    def test_full_stack_cli_wins_over_everything(self, cwd, monkeypatch, _isolated_home) -> None:
        _clear_cellsense_env(monkeypatch)
        home_dir = _isolated_home / ".cellsense"
        home_dir.mkdir()
        (home_dir / "config.toml").write_text('[model]\ndefault = "home-model"\n')
        local_dir = cwd / ".cellsense"
        local_dir.mkdir()
        (local_dir / "config.toml").write_text('[model]\ndefault = "local-model"\n')
        monkeypatch.setenv("CELLSENSE_MODEL_DEFAULT", "env-model")
        config = load_config({"model": {"default": "cli-model"}}, cwd=cwd)
        assert config.model.default == "cli-model"

    def test_unrelated_fields_still_come_from_lower_precedence_levels(
        self, cwd, monkeypatch, _isolated_home
    ) -> None:
        _clear_cellsense_env(monkeypatch)
        local_dir = cwd / ".cellsense"
        local_dir.mkdir()
        (local_dir / "config.toml").write_text('[permissions]\nmode = "allow"\n')
        config = load_config({"model": {"default": "cli-model"}}, cwd=cwd)
        assert config.model.default == "cli-model"
        assert config.permissions.mode == "allow"


class TestEnvCoercion:
    def test_int_field_is_coerced(self, cwd, monkeypatch) -> None:
        _clear_cellsense_env(monkeypatch)
        monkeypatch.setenv("CELLSENSE_AGENT_MAX_TOOL_ROUNDS", "5")
        config = load_config({}, cwd=cwd)
        assert config.agent.max_tool_rounds == 5
        assert isinstance(config.agent.max_tool_rounds, int)

    def test_bool_field_is_coerced(self, cwd, monkeypatch) -> None:
        _clear_cellsense_env(monkeypatch)
        monkeypatch.setenv("CELLSENSE_AGENT_GUARDRAIL", "false")
        config = load_config({}, cwd=cwd)
        assert config.agent.guardrail is False

    def test_list_field_is_coerced_from_a_comma_separated_string(self, cwd, monkeypatch) -> None:
        _clear_cellsense_env(monkeypatch)
        monkeypatch.setenv("CELLSENSE_PERMISSIONS_ALLOW", "aggregate, plot")
        config = load_config({}, cwd=cwd)
        assert config.permissions.allow == ["aggregate", "plot"]


class TestErrors:
    def test_malformed_toml_raises_config_error(self, cwd) -> None:
        local_dir = cwd / ".cellsense"
        local_dir.mkdir()
        (local_dir / "config.toml").write_text("this is not [valid toml")
        with pytest.raises(ConfigError, match="Malformed TOML"):
            load_config({}, cwd=cwd)

    def test_unknown_top_level_key_raises_with_close_match_hint(self, cwd) -> None:
        local_dir = cwd / ".cellsense"
        local_dir.mkdir()
        (local_dir / "config.toml").write_text('[modle]\ndefault = "x"\n')
        with pytest.raises(ConfigError) as exc_info:
            load_config({}, cwd=cwd)
        assert "model" in (exc_info.value.hint or "")

    def test_unknown_nested_key_raises_with_close_match_hint(self, cwd) -> None:
        local_dir = cwd / ".cellsense"
        local_dir.mkdir()
        (local_dir / "config.toml").write_text('[model]\ndefaultt = "x"\n')
        with pytest.raises(ConfigError) as exc_info:
            load_config({}, cwd=cwd)
        assert "default" in (exc_info.value.hint or "")

    def test_unknown_key_via_cli_overrides_also_raises(self, cwd) -> None:
        with pytest.raises(ConfigError):
            load_config({"model": {"defaultt": "x"}}, cwd=cwd)


class TestRedaction:
    def test_redact_masks_key_like_field_names(self) -> None:
        out = _redact({"api_key": "sk-secret", "nested": {"auth_token": "abc"}})
        assert out["api_key"] == "***redacted***"
        assert out["nested"]["auth_token"] == "***redacted***"

    def test_redact_leaves_non_sensitive_fields_untouched(self) -> None:
        out = _redact({"model": "gpt-4o", "temperature": 0.0})
        assert out == {"model": "gpt-4o", "temperature": 0.0}

    def test_redact_recurses_into_lists(self) -> None:
        out = _redact({"items": [{"secret": "x"}, {"value": 1}]})
        assert out["items"][0]["secret"] == "***redacted***"
        assert out["items"][1]["value"] == 1

    def test_config_redacted_dict_does_not_raise_and_is_plain_dict(self) -> None:
        out = Config().redacted_dict()
        assert isinstance(out, dict)
        assert out["model"]["default"] == Config().model.default

    def test_redact_matches_key_pattern_case_insensitively(self) -> None:
        out = _redact({"API_KEY": "shh", "Password": "shh", "Secret": "shh"})
        assert all(v == "***redacted***" for v in out.values())


class TestModelConfigToSettings:
    def test_maps_fields_onto_model_settings(self) -> None:
        settings = ModelConfig(
            temperature=0.5, max_tokens=1000, timeout_s=30, max_retries=1, streaming=True
        ).to_settings()
        assert settings.temperature == 0.5
        assert settings.max_tokens == 1000
        assert settings.timeout_s == 30
        assert settings.max_retries == 1
        assert settings.streaming is True


class TestWriteDefaultConfig:
    def test_writes_the_canonical_toml_and_creates_parents(self, tmp_path) -> None:
        target = tmp_path / "nested" / "config.toml"
        write_default_config(target)
        assert target.is_file()
        content = target.read_text()
        assert "[model]" in content
        assert 'default = "groq:openai/gpt-oss-120b"' in content

    def test_written_config_round_trips_through_load_config(
        self, tmp_path, monkeypatch, cwd
    ) -> None:
        _clear_cellsense_env(monkeypatch)
        local_dir = cwd / ".cellsense"
        write_default_config(local_dir / "config.toml")
        config = load_config({}, cwd=cwd)
        assert config == Config()  # canonical defaults round-trip to the same Config
