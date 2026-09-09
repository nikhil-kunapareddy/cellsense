"""Exception hierarchy for CellSense.

Every error raised deliberately by CellSense derives from :class:`CellSenseError`
so the CLI can render it as a clean message instead of a traceback. Anything that
escapes as a bare ``Exception`` is a bug and is reported as such.
"""

from __future__ import annotations


class CellSenseError(Exception):
    """Base class for all CellSense errors.

    Attributes:
        hint: Optional actionable next step shown to the user beneath the message.
    """

    exit_code: int = 1

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


# ── Configuration ──────────────────────────────────────────────────────────────


class ConfigError(CellSenseError):
    """Malformed config file, bad CLI flag combination, or unusable settings."""

    exit_code = 2


class MissingAPIKeyError(ConfigError):
    """No API key found for the selected provider."""

    def __init__(self, provider: str, env_key: str) -> None:
        super().__init__(
            f"No API key for provider {provider!r}.",
            hint=f"Set {env_key} in your environment or in a .env file.",
        )
        self.provider = provider
        self.env_key = env_key


class MissingDependencyError(ConfigError):
    """A provider's optional integration package is not installed."""

    def __init__(self, provider: str, package: str, extra: str) -> None:
        super().__init__(
            f"Provider {provider!r} requires the {package!r} package.",
            hint=f"Install it with:  pip install 'cellsense[{extra}]'",
        )
        self.provider = provider
        self.package = package
        self.extra = extra


# ── Providers ──────────────────────────────────────────────────────────────────


class ProviderError(CellSenseError):
    """The upstream model API failed."""

    exit_code = 3


class UnknownProviderError(ProviderError):
    def __init__(self, name: str, known: list[str]) -> None:
        super().__init__(
            f"Unknown provider {name!r}.",
            hint=f"Known providers: {', '.join(known)}",
        )


class AuthenticationError(ProviderError):
    """The API key was rejected."""


class RateLimitError(ProviderError):
    """Rate limited or out of quota, after retries were exhausted."""


class ModelRefusedError(ProviderError):
    """The model returned a refusal or an unusable/empty response."""


# ── Data ───────────────────────────────────────────────────────────────────────


class DataError(CellSenseError):
    """A workspace file could not be loaded or understood."""

    exit_code = 4


class UnsupportedFileTypeError(DataError):
    def __init__(self, path: str, suffix: str, supported: set[str]) -> None:
        super().__init__(
            f"Unsupported file type {suffix!r}: {path}",
            hint=f"Supported extensions: {', '.join(sorted(supported))}",
        )


# ── Tools ──────────────────────────────────────────────────────────────────────


class ToolError(CellSenseError):
    """A tool failed. Recoverable: the message is fed back to the model."""


class ApprovalDeniedError(CellSenseError):
    """The user declined a permission-gated tool call."""


# ── Control flow ───────────────────────────────────────────────────────────────


class TurnCancelledError(CellSenseError):
    """The user interrupted an in-flight turn (Esc / Ctrl-C)."""

    exit_code = 130
