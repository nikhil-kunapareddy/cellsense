"""Pins cellsense.providers.base: translate_provider_error's status/class-name
mapping and build_chat_model's lazy-import + error-translation pipeline.
"""

from __future__ import annotations

import pytest

from cellsense.errors import (
    AuthenticationError,
    ConfigError,
    MissingAPIKeyError,
    MissingDependencyError,
    ProviderError,
    RateLimitError,
)
from cellsense.providers.base import ModelSettings, build_chat_model, translate_provider_error


class _FakeProviderError(Exception):
    pass


class TestTranslateProviderError:
    def test_status_401_is_authentication_error(self) -> None:
        exc = _FakeProviderError("nope")
        exc.status_code = 401
        assert isinstance(translate_provider_error(exc, "anthropic"), AuthenticationError)

    def test_status_403_is_authentication_error(self) -> None:
        exc = _FakeProviderError("nope")
        exc.status_code = 403
        assert isinstance(translate_provider_error(exc, "anthropic"), AuthenticationError)

    def test_status_429_is_rate_limit_error(self) -> None:
        exc = _FakeProviderError("slow down")
        exc.status_code = 429
        assert isinstance(translate_provider_error(exc, "groq"), RateLimitError)

    def test_other_status_is_generic_provider_error(self) -> None:
        exc = _FakeProviderError("boom")
        exc.status_code = 500
        result = translate_provider_error(exc, "openai")
        assert isinstance(result, ProviderError)
        assert not isinstance(result, (AuthenticationError, RateLimitError))

    def test_status_nested_under_response_attribute(self) -> None:
        class _Response:
            status_code = 401

        exc = _FakeProviderError("nope")
        exc.response = _Response()
        assert isinstance(translate_provider_error(exc, "openai"), AuthenticationError)

    def test_status_field_named_status_not_status_code(self) -> None:
        exc = _FakeProviderError("slow")
        exc.status = 429
        assert isinstance(translate_provider_error(exc, "groq"), RateLimitError)

    def test_class_name_hints_authentication_without_a_status(self) -> None:
        class _FakeAuthenticationError(Exception):
            pass

        assert isinstance(
            translate_provider_error(_FakeAuthenticationError("x"), "gemini"), AuthenticationError
        )

    def test_class_name_hints_permission_denied_as_authentication(self) -> None:
        class PermissionDeniedError(Exception):
            pass

        assert isinstance(
            translate_provider_error(PermissionDeniedError("x"), "gemini"), AuthenticationError
        )

    def test_class_name_hints_rate_limit_without_a_status(self) -> None:
        class _FakeRateLimitError(Exception):
            pass

        assert isinstance(
            translate_provider_error(_FakeRateLimitError("x"), "groq"), RateLimitError
        )

    def test_unparseable_status_falls_back_to_class_name_sniffing(self) -> None:
        exc = _FakeProviderError("weird")
        exc.status_code = "not-a-number"
        result = translate_provider_error(exc, "openai")
        assert isinstance(result, ProviderError)
        assert not isinstance(result, (AuthenticationError, RateLimitError))

    def test_message_includes_provider_name(self) -> None:
        result = translate_provider_error(_FakeProviderError("boom"), "anthropic")
        assert "anthropic" in result.message


class TestBuildChatModel:
    def test_missing_api_key_raised_before_loader_is_called(self, monkeypatch) -> None:
        monkeypatch.delenv("FAKE_API_KEY", raising=False)
        calls = []

        def _loader():
            calls.append("called")
            raise AssertionError("loader must not be called when the API key is missing")

        with pytest.raises(MissingAPIKeyError) as exc_info:
            build_chat_model(
                provider="fake",
                env_key="FAKE_API_KEY",
                package="fake_pkg",
                extra="fake",
                loader=_loader,
            )
        assert calls == []
        assert exc_info.value.env_key == "FAKE_API_KEY"
        assert exc_info.value.provider == "fake"

    def test_loader_import_error_becomes_missing_dependency_error(self, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_API_KEY", "secret")

        def _loader():
            raise ImportError("no such package")

        with pytest.raises(MissingDependencyError) as exc_info:
            build_chat_model(
                provider="fake",
                env_key="FAKE_API_KEY",
                package="fake_pkg",
                extra="fake",
                loader=_loader,
            )
        assert exc_info.value.package == "fake_pkg"
        assert exc_info.value.extra == "fake"

    def test_constructor_generic_exception_is_translated(self, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_API_KEY", "secret")

        class _FakeModel:
            def __init__(self, **kwargs):
                raise _FakeProviderError("bad key format")

        with pytest.raises(ProviderError):
            build_chat_model(
                provider="fake",
                env_key="FAKE_API_KEY",
                package="fake_pkg",
                extra="fake",
                loader=lambda: _FakeModel,
            )

    def test_constructor_cellsense_error_is_reraised_unchanged(self, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_API_KEY", "secret")
        original = ConfigError("custom config problem")

        class _FakeModel:
            def __init__(self, **kwargs):
                raise original

        with pytest.raises(ConfigError) as exc_info:
            build_chat_model(
                provider="fake",
                env_key="FAKE_API_KEY",
                package="fake_pkg",
                extra="fake",
                loader=lambda: _FakeModel,
            )
        assert exc_info.value is original

    def test_successful_build_returns_the_constructed_model(self, monkeypatch) -> None:
        monkeypatch.setenv("FAKE_API_KEY", "secret")

        class _FakeModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        result = build_chat_model(
            provider="fake",
            env_key="FAKE_API_KEY",
            package="fake_pkg",
            extra="fake",
            loader=lambda: _FakeModel,
            model="some-model",
            temperature=0.0,
        )
        assert isinstance(result, _FakeModel)
        assert result.kwargs == {"model": "some-model", "temperature": 0.0}


def test_model_settings_defaults() -> None:
    settings = ModelSettings()
    assert settings.temperature == 0.0
    assert settings.max_tokens == 4096
    assert settings.max_retries == 3
    assert settings.streaming is False
