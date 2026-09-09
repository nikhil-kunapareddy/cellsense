"""Pins cellsense.errors: the CellSenseError hierarchy, exit codes, and the
hint-carrying constructors of the domain-specific error subclasses.
"""

from __future__ import annotations

from cellsense.errors import (
    ApprovalDeniedError,
    AuthenticationError,
    CellSenseError,
    ConfigError,
    DataError,
    MissingAPIKeyError,
    MissingDependencyError,
    ModelRefusedError,
    ProviderError,
    RateLimitError,
    ToolError,
    TurnCancelledError,
    UnknownProviderError,
    UnsupportedFileTypeError,
)


def test_base_error_carries_message_and_optional_hint() -> None:
    err = CellSenseError("boom", hint="try again")
    assert err.message == "boom"
    assert err.hint == "try again"
    assert str(err) == "boom"


def test_hint_defaults_to_none() -> None:
    assert CellSenseError("boom").hint is None


def test_default_exit_code_is_one() -> None:
    assert CellSenseError("boom").exit_code == 1


def test_subclass_exit_codes() -> None:
    assert ConfigError("x").exit_code == 2
    assert ProviderError("x").exit_code == 3
    assert DataError("x").exit_code == 4
    assert TurnCancelledError("x").exit_code == 130
    assert ToolError("x").exit_code == 1  # inherits the base default


def test_missing_api_key_error_names_provider_and_env_key() -> None:
    err = MissingAPIKeyError("anthropic", "ANTHROPIC_API_KEY")
    assert err.provider == "anthropic"
    assert err.env_key == "ANTHROPIC_API_KEY"
    assert "anthropic" in err.message
    assert "ANTHROPIC_API_KEY" in (err.hint or "")
    assert isinstance(err, ConfigError)


def test_missing_dependency_error_names_package_and_extra() -> None:
    err = MissingDependencyError("groq", "langchain_groq", "groq")
    assert err.package == "langchain_groq"
    assert err.extra == "groq"
    assert "langchain_groq" in err.message
    assert "cellsense[groq]" in (err.hint or "")


def test_unknown_provider_error_lists_known_providers() -> None:
    err = UnknownProviderError("bogus", ["anthropic", "groq"])
    assert "bogus" in err.message
    assert "anthropic" in (err.hint or "") and "groq" in (err.hint or "")


def test_unsupported_file_type_error_names_suffix_and_supported_set() -> None:
    err = UnsupportedFileTypeError("data.json", ".json", {".csv", ".xlsx"})
    assert ".json" in err.message
    assert ".csv" in (err.hint or "")


def test_error_hierarchy() -> None:
    assert issubclass(MissingAPIKeyError, ConfigError)
    assert issubclass(MissingDependencyError, ConfigError)
    assert issubclass(UnknownProviderError, ProviderError)
    assert issubclass(AuthenticationError, ProviderError)
    assert issubclass(RateLimitError, ProviderError)
    assert issubclass(ModelRefusedError, ProviderError)
    assert issubclass(UnsupportedFileTypeError, DataError)
    for cls in (
        ConfigError,
        ProviderError,
        DataError,
        ToolError,
        ApprovalDeniedError,
        TurnCancelledError,
    ):
        assert issubclass(cls, CellSenseError)
