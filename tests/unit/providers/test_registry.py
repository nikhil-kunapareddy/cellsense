"""Pins cellsense.providers.registry: selector parsing (bare provider,
provider:model, bare model id, unknown-id synthesis, ambiguity), cost
estimation, and available_providers().
"""

from __future__ import annotations

import pytest

from cellsense.errors import ConfigError, UnknownProviderError
from cellsense.providers import registry as reg
from cellsense.providers.base import ModelSettings, ModelSpec, ProviderSpec

SETTINGS = ModelSettings()


class TestResolveModel:
    def test_none_selector_uses_the_documented_default(self) -> None:
        resolved = reg.resolve_model(None, SETTINGS)
        assert resolved.provider_spec.name == "groq"
        assert resolved.model_spec.id == reg.DEFAULT_MODEL_SELECTOR.split(":", 1)[1]

    def test_bare_provider_uses_its_default_model(self) -> None:
        resolved = reg.resolve_model("anthropic", SETTINGS)
        assert resolved.provider_spec.name == "anthropic"
        assert resolved.model_spec.id == reg.PROVIDERS["anthropic"].default_model

    def test_explicit_provider_and_known_model_id(self) -> None:
        resolved = reg.resolve_model("openai:gpt-4o-mini", SETTINGS)
        assert resolved.provider_spec.name == "openai"
        assert resolved.model_spec.id == "gpt-4o-mini"
        assert resolved.model_spec.input_usd_per_mtok is not None

    def test_explicit_provider_with_unknown_model_id_synthesizes_a_spec(self) -> None:
        resolved = reg.resolve_model("anthropic:some-new-model-nobody-has-heard-of", SETTINGS)
        assert resolved.model_spec.id == "some-new-model-nobody-has-heard-of"
        assert resolved.model_spec.provider == "anthropic"
        assert resolved.model_spec.input_usd_per_mtok is None
        assert resolved.model_spec.output_usd_per_mtok is None
        assert resolved.model_spec.context_window == 0

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(UnknownProviderError):
            reg.resolve_model("not_a_real_provider:some-model", SETTINGS)

    def test_unknown_provider_error_lists_known_providers(self) -> None:
        with pytest.raises(UnknownProviderError) as exc_info:
            reg.resolve_model("bogus:x", SETTINGS)
        for name in reg.PROVIDERS:
            assert name in (exc_info.value.hint or "")

    def test_bare_model_id_unique_across_providers_resolves(self) -> None:
        resolved = reg.resolve_model("gpt-4o", SETTINGS)
        assert resolved.provider_spec.name == "openai"
        assert resolved.model_spec.id == "gpt-4o"

    def test_bare_unknown_model_or_provider_raises_config_error(self) -> None:
        with pytest.raises(ConfigError, match="Unknown model or provider"):
            reg.resolve_model("totally-made-up-id", SETTINGS)

    def test_ambiguous_bare_model_id_raises_config_error_listing_candidates(
        self, monkeypatch
    ) -> None:
        shared_model = ModelSpec(
            id="shared-model",
            provider="fake-a",
            display="Shared A",
            context_window=1000,
            max_output_tokens=100,
            input_usd_per_mtok=None,
            output_usd_per_mtok=None,
        )
        provider_a = ProviderSpec(
            name="fake-a",
            label="Fake A",
            env_key="FAKE_A_KEY",
            package="pkg_a",
            extra="a",
            default_model="shared-model",
            models=(shared_model,),
            builder=lambda *a, **k: None,
        )
        shared_model_b = ModelSpec(
            id="shared-model",
            provider="fake-b",
            display="Shared B",
            context_window=1000,
            max_output_tokens=100,
            input_usd_per_mtok=None,
            output_usd_per_mtok=None,
        )
        provider_b = ProviderSpec(
            name="fake-b",
            label="Fake B",
            env_key="FAKE_B_KEY",
            package="pkg_b",
            extra="b",
            default_model="shared-model",
            models=(shared_model_b,),
            builder=lambda *a, **k: None,
        )
        monkeypatch.setattr(reg, "PROVIDERS", {"fake-a": provider_a, "fake-b": provider_b})

        with pytest.raises(ConfigError) as exc_info:
            reg.resolve_model("shared-model", SETTINGS)
        assert "fake-a:shared-model" in (exc_info.value.hint or "")
        assert "fake-b:shared-model" in (exc_info.value.hint or "")


class TestAvailableProviders:
    def test_only_providers_with_a_set_env_key_are_available(self, monkeypatch) -> None:
        for spec in reg.PROVIDERS.values():
            monkeypatch.delenv(spec.env_key, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        monkeypatch.setenv("GROQ_API_KEY", "y")
        assert reg.available_providers() == ["anthropic", "groq"]

    def test_no_keys_set_yields_empty_list(self, monkeypatch) -> None:
        for spec in reg.PROVIDERS.values():
            monkeypatch.delenv(spec.env_key, raising=False)
        assert reg.available_providers() == []


class TestEstimateCost:
    def test_known_pricing_matches_manual_calculation(self) -> None:
        spec = reg.PROVIDERS["openai"].models[0]  # gpt-4o: 2.50 / 10.00 per mtok
        cost = reg.estimate_cost(spec, input_tokens=1_000_000, output_tokens=500_000)
        assert cost == pytest.approx(2.50 * 1 + 10.00 * 0.5)

    def test_unpriced_model_returns_none(self) -> None:
        compound = next(m for m in reg.PROVIDERS["groq"].models if m.id == "groq/compound")
        assert reg.estimate_cost(compound, 1000, 1000) is None

    def test_zero_tokens_costs_zero_for_a_priced_model(self) -> None:
        spec = reg.PROVIDERS["openai"].models[0]
        assert reg.estimate_cost(spec, 0, 0) == 0.0


def test_list_models_flattens_every_provider_curated_list() -> None:
    expected = sum(len(spec.models) for spec in reg.PROVIDERS.values())
    assert len(reg.list_models()) == expected


def test_providers_dict_is_keyed_by_provider_name() -> None:
    for name, spec in reg.PROVIDERS.items():
        assert spec.name == name


def test_resolved_model_chat_model_is_built_lazily() -> None:
    """Merely resolving a selector must not build (or import an SDK for) the
    chat model -- only touching .chat_model does.
    """
    resolved = reg.resolve_model("anthropic", SETTINGS)
    assert resolved._chat_model is None  # not built yet, by construction
