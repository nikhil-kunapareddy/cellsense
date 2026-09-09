"""The provider directory: selector parsing, cost estimation, and model listing.

This module imports all five provider modules at top level -- that is fine and
does not violate the lazy-import rule, because the provider modules themselves
only import their ``langchain_*`` package lazily inside ``build()``. Importing
``cellsense.providers.registry`` never pulls in any SDK.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cellsense.errors import ConfigError, UnknownProviderError
from cellsense.providers import anthropic, gemini, groq, llama, openai
from cellsense.providers.base import ModelSettings, ModelSpec, ProviderSpec

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

__all__ = [
    "DEFAULT_MODEL_SELECTOR",
    "PROVIDERS",
    "ResolvedModel",
    "available_providers",
    "estimate_cost",
    "list_models",
    "resolve_model",
]

PROVIDERS: dict[str, ProviderSpec] = {
    spec.name: spec for spec in (anthropic.SPEC, openai.SPEC, gemini.SPEC, groq.SPEC, llama.SPEC)
}

# docs/ARCHITECTURE.md §5's sample ``[model].default`` is "groq:llama-3.3-70b-versatile"
# -- that id is no longer served by Groq as of this writing (see groq.py's module
# docstring for the live-call evidence), so this constant points at the verified
# live replacement instead of a selector that would fail on first use. Used
# whenever ``resolve_model`` is called with ``selector=None`` and the caller has
# not already substituted a config-resolved default (e.g. Config.model.default).
DEFAULT_MODEL_SELECTOR = "groq:openai/gpt-oss-120b"


@dataclass
class ResolvedModel:
    """The outcome of resolving a ``--model`` selector.

    ``chat_model`` is built lazily on first access (not at resolution time) so
    that merely *resolving* a selector -- e.g. for ``/model`` or ``--debug``
    dumps -- never requires an API key or imports a provider SDK.
    """

    provider_spec: ProviderSpec
    model_spec: ModelSpec
    settings: ModelSettings
    _chat_model: BaseChatModel | None = field(default=None, init=False, repr=False)

    @property
    def chat_model(self) -> BaseChatModel:
        if self._chat_model is None:
            self._chat_model = self.provider_spec.build(self.model_spec.id, self.settings)
        return self._chat_model


def _get_provider(name: str) -> ProviderSpec:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise UnknownProviderError(name, sorted(PROVIDERS)) from None


def _lookup_or_synthesize(provider_spec: ProviderSpec, model_id: str) -> ModelSpec:
    """Find ``model_id`` in the provider's curated list, or synthesize a spec with
    unknown pricing/limits. Per ARCHITECTURE.md §4 rule 2, the curated list is
    metadata, not a whitelist -- an unrecognized id must still resolve.
    """
    for model_spec in provider_spec.models:
        if model_spec.id == model_id:
            return model_spec
    return ModelSpec(
        id=model_id,
        provider=provider_spec.name,
        display=model_id,
        # 0 signals "unknown" for a synthesized spec -- we have no curated data
        # for this id and refuse to guess a context window or output cap.
        context_window=0,
        max_output_tokens=0,
        input_usd_per_mtok=None,
        output_usd_per_mtok=None,
    )


def resolve_model(selector: str | None, settings: ModelSettings) -> ResolvedModel:
    """Resolve a ``--model`` selector to a :class:`ResolvedModel`.

    Selector syntax (ARCHITECTURE.md §4 rule 3):

    * ``"provider:model_id"`` -- explicit. Unknown provider raises
      :class:`UnknownProviderError`; unknown model_id synthesizes a spec.
    * ``"provider"`` (bare, matches a key in :data:`PROVIDERS`) -- uses that
      provider's ``default_model``.
    * ``"model_id"`` (bare, matches no provider name) -- searched across every
      provider's curated ``models`` tuple. Exactly one match resolves; zero or
      multiple matches raise :class:`ConfigError` (the latter lists candidates).
    * ``None`` -- falls back to :data:`DEFAULT_MODEL_SELECTOR`.
    """
    sel = selector if selector is not None else DEFAULT_MODEL_SELECTOR

    if ":" in sel:
        provider_name, model_id = sel.split(":", 1)
        provider_spec = _get_provider(provider_name)
        model_spec = _lookup_or_synthesize(provider_spec, model_id)
        return ResolvedModel(provider_spec, model_spec, settings)

    if sel in PROVIDERS:
        provider_spec = PROVIDERS[sel]
        model_spec = _lookup_or_synthesize(provider_spec, provider_spec.default_model)
        return ResolvedModel(provider_spec, model_spec, settings)

    candidates = [
        (provider_spec, model_spec)
        for provider_spec in PROVIDERS.values()
        for model_spec in provider_spec.models
        if model_spec.id == sel
    ]
    if len(candidates) == 1:
        provider_spec, model_spec = candidates[0]
        return ResolvedModel(provider_spec, model_spec, settings)
    if len(candidates) > 1:
        names = ", ".join(f"{p.name}:{m.id}" for p, m in candidates)
        raise ConfigError(
            f"Ambiguous model id {sel!r} matches multiple providers.",
            hint=f"Qualify it as one of: {names}",
        )

    raise ConfigError(
        f"Unknown model or provider {sel!r}.",
        hint=f"Known providers: {', '.join(sorted(PROVIDERS))}",
    )


def available_providers() -> list[str]:
    """Providers whose env key is currently set -- drives the startup message and
    ``/model``'s "ready to use" list.
    """
    return [name for name, spec in PROVIDERS.items() if os.environ.get(spec.env_key)]


def estimate_cost(model_spec: ModelSpec, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate USD cost for a call, or ``None`` when either price is unknown."""
    if model_spec.input_usd_per_mtok is None or model_spec.output_usd_per_mtok is None:
        return None
    return (
        input_tokens * model_spec.input_usd_per_mtok
        + output_tokens * model_spec.output_usd_per_mtok
    ) / 1_000_000


def list_models() -> list[ModelSpec]:
    """Every curated model across every provider, for ``/model`` and ``cellsense
    models``. Does not include synthesized (unrecognized) ids -- those only exist
    once resolved.
    """
    return [
        model_spec for provider_spec in PROVIDERS.values() for model_spec in provider_spec.models
    ]
