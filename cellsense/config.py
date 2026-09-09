"""Settings resolution: CLI flags > env > ./.cellsense/config.toml >
~/.cellsense/config.toml > built-in defaults (ARCHITECTURE.md §5).

``load_config`` is the single entry point. Everything else in this module is a
private helper it composes: TOML parsing, environment-variable overlay, deep
merge, and unknown-key validation with a "did you mean" hint.
"""

from __future__ import annotations

import difflib
import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from cellsense.errors import ConfigError
from cellsense.providers.base import ModelSettings

__all__ = [
    "AgentConfig",
    "Config",
    "ModelConfig",
    "PermissionsConfig",
    "TelemetryConfig",
    "UiConfig",
    "load_config",
    "write_default_config",
]

_SENSITIVE_FIELD_PATTERN = ("key", "token", "secret", "password", "authorization")

# The canonical default TOML, also used by write_default_config(). Kept as a plain
# string (not generated from dataclass defaults) so `cellsense init` produces a
# file with comments and the exact layout documented in ARCHITECTURE.md §5.
#
# DEVIATION: ARCHITECTURE.md §5's sample uses `default = "groq:llama-3.3-70b-versatile"`.
# That model id is no longer served by Groq as of this writing (see
# providers/groq.py's module docstring) -- pointing new configs at a dead model
# by default would fail on first use, so this mirrors
# providers.registry.DEFAULT_MODEL_SELECTOR instead.
_DEFAULT_TOML = """\
[model]
default = "groq:openai/gpt-oss-120b"
temperature = 0.0
max_tokens = 4096
timeout_s = 120
max_retries = 3

[agent]
max_tool_rounds = 8
max_subtasks = 4
planner = true          # false => always single-shot, skips the planner call
guardrail = true

[permissions]
mode = "prompt"         # "prompt" | "allow" | "deny"
allow = ["aggregate", "filter_rows", "join", "describe"]

[ui]
theme = "auto"
show_tool_args = true

[telemetry]
log_dir = "~/.cellsense/logs"
level = "info"
"""


# ── Config tree ──────────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    default: str = "groq:openai/gpt-oss-120b"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: int = 120
    max_retries: int = 3
    streaming: bool = False

    def to_settings(self) -> ModelSettings:
        """Project this section onto the :class:`ModelSettings` the providers layer
        actually consumes -- keeps ``Config`` (a TOML-shaped tree) and
        ``ModelSettings`` (a providers-shaped value object) independently evolvable.
        """
        return ModelSettings(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            streaming=self.streaming,
        )


@dataclass
class AgentConfig:
    max_tool_rounds: int = 8
    max_subtasks: int = 4
    planner: bool = True
    guardrail: bool = True


@dataclass
class PermissionsConfig:
    mode: Literal["prompt", "allow", "deny"] = "prompt"
    allow: list[str] = field(
        default_factory=lambda: ["aggregate", "filter_rows", "join", "describe"]
    )


@dataclass
class UiConfig:
    theme: str = "auto"
    show_tool_args: bool = True


@dataclass
class TelemetryConfig:
    log_dir: str = "~/.cellsense/logs"
    level: str = "info"


@dataclass
class Config:
    """The fully-resolved, precedence-applied settings tree."""

    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    def redacted_dict(self) -> dict[str, Any]:
        """A plain-dict rendering safe to print in ``--debug`` dumps.

        ``Config`` never actually holds API keys (those live in the environment
        and are read by the providers layer at build time), so there is nothing
        to strip in practice -- this walks the tree defensively anyway and masks
        any field whose *name* looks sensitive, in case a future section adds one.
        """
        return _redact(asdict(self))


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, val in value.items():
            if any(marker in key.lower() for marker in _SENSITIVE_FIELD_PATTERN):
                out[key] = "***redacted***"
            else:
                out[key] = _redact(val)
        return out
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


# ── Defaults schema (for validation + merge) ─────────────────────────────────────


def _defaults_dict() -> dict[str, Any]:
    return asdict(Config())


def _closest_key(bad_key: str, valid_keys: list[str]) -> str | None:
    matches = difflib.get_close_matches(bad_key, valid_keys, n=1)
    return matches[0] if matches else None


def _validate_keys(data: dict[str, Any], schema: dict[str, Any], *, path: str = "") -> None:
    """Raise :class:`ConfigError` naming the closest valid key for any key in
    ``data`` that doesn't exist in ``schema`` at the same nesting level.
    """
    for key, value in data.items():
        qualified = f"{path}.{key}" if path else key
        if key not in schema:
            hint_key = _closest_key(key, list(schema.keys()))
            hint = (
                f"Did you mean {hint_key!r}?"
                if hint_key
                else f"Valid keys: {', '.join(schema.keys())}"
            )
            raise ConfigError(f"Unknown config key {qualified!r}.", hint=hint)
        if isinstance(value, dict) and isinstance(schema[key], dict):
            _validate_keys(value, schema[key], path=qualified)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        try:
            return tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Malformed TOML in {path}: {exc}") from exc


def _coerce_env_value(raw: str, template: Any) -> Any:
    if isinstance(template, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(template, int):
        return int(raw)
    if isinstance(template, float):
        return float(raw)
    if isinstance(template, list):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw


def _env_overrides(schema: dict[str, Any]) -> dict[str, Any]:
    """Overlay env vars named ``CELLSENSE_<SECTION>_<KEY>`` onto the config tree,
    e.g. ``CELLSENSE_MODEL_DEFAULT`` -> ``model.default``,
    ``CELLSENSE_PERMISSIONS_MODE`` -> ``permissions.mode``. Values are coerced to
    match the type of the built-in default for that key.
    """
    overrides: dict[str, Any] = {}
    for section, keys in schema.items():
        for key, template in keys.items():
            env_name = f"CELLSENSE_{section.upper()}_{key.upper()}"
            raw = os.environ.get(env_name)
            if raw is None:
                continue
            overrides.setdefault(section, {})[key] = _coerce_env_value(raw, template)
    return overrides


def _build_config(data: dict[str, Any]) -> Config:
    return Config(
        model=ModelConfig(**data.get("model", {})),
        agent=AgentConfig(**data.get("agent", {})),
        permissions=PermissionsConfig(**data.get("permissions", {})),
        ui=UiConfig(**data.get("ui", {})),
        telemetry=TelemetryConfig(**data.get("telemetry", {})),
    )


def load_config(cli_overrides: dict[str, Any], *, cwd: Path) -> Config:
    """Resolve settings with precedence: CLI > env > ./.cellsense/config.toml >
    ~/.cellsense/config.toml > built-in defaults.

    ``cli_overrides`` is a nested dict mirroring the TOML shape, e.g.
    ``{"model": {"default": "claude:claude-sonnet-5"}}`` -- built by ``cli.py``
    from whichever flags the user actually passed (absent flags should be omitted
    entirely, not passed as ``None``, so they don't shadow a lower-precedence
    value). Also loads ``.env`` from ``cwd`` before resolving env overrides, so
    provider API keys and ``CELLSENSE_*`` overrides in a local .env take effect.
    """
    load_dotenv(cwd / ".env")

    schema = _defaults_dict()
    merged = schema

    home_toml = Path.home() / ".cellsense" / "config.toml"
    if home_toml.is_file():
        home_data = _load_toml(home_toml)
        _validate_keys(home_data, schema)
        merged = _deep_merge(merged, home_data)

    local_toml = cwd / ".cellsense" / "config.toml"
    if local_toml.is_file():
        local_data = _load_toml(local_toml)
        _validate_keys(local_data, schema)
        merged = _deep_merge(merged, local_data)

    merged = _deep_merge(merged, _env_overrides(schema))

    if cli_overrides:
        _validate_keys(cli_overrides, schema)
        merged = _deep_merge(merged, cli_overrides)

    return _build_config(merged)


def write_default_config(path: Path) -> None:
    """Write the canonical default config to ``path`` (parents created as needed).
    Used by the future ``cellsense init`` command.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFAULT_TOML, encoding="utf-8")
