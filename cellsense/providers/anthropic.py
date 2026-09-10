"""Anthropic provider: langchain_anthropic.ChatAnthropic.

Pricing and context/output limits below are the Claude 5 generation figures current
as of this writing. ``claude-opus-5`` is the provider default -- the most capable
model in the family, and the one Anthropic's own guidance says to reach for unless
the caller names another. Downgrading for cost is the user's call, made with
``--model anthropic:claude-sonnet-5``.

**Sampling parameters are not universal on this API.** The Claude 5 generation
(Fable 5.x, Opus 5, Sonnet 5) and Opus 4.7/4.8 removed ``temperature`` /
``top_p`` / ``top_k`` and reject them with a 400. This is easy to miss, because
``langchain_anthropic`` does not drop them for you: ``anthropic>=1`` stopped
accepting them as named arguments, so ``_sdk_compat._route_unsupported_sampling_params``
relocates them into ``extra_body``, which the SDK merges into the request JSON
verbatim -- "the wire payload is identical on both majors", per that module's own
docstring. So a ``temperature=0.0`` that looks absent from the payload dict is
still sent, and still 400s. ``_build`` therefore omits it per model, driven by
:data:`_ACCEPTS_SAMPLING_PARAMS`. (Fable models are the one case the integration
catches itself, raising ``ValueError`` at *request* time -- past
``build_chat_model``'s error translation, so it would surface as a raw traceback
rather than a ``ProviderError``.)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from cellsense.providers.base import ModelSettings, ModelSpec, ProviderSpec, build_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

PROVIDER = "anthropic"
ENV_KEY = "ANTHROPIC_API_KEY"
PACKAGE = "langchain_anthropic"
EXTRA = "anthropic"
DEFAULT_MODEL = "claude-opus-5"

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="claude-fable-5-1",
        provider=PROVIDER,
        display="Claude Fable 5.1",
        context_window=1_000_000,
        max_output_tokens=128_000,
        input_usd_per_mtok=10.00,
        output_usd_per_mtok=50.00,
    ),
    ModelSpec(
        id="claude-opus-5",
        provider=PROVIDER,
        display="Claude Opus 5",
        context_window=1_000_000,
        max_output_tokens=128_000,
        input_usd_per_mtok=5.00,
        output_usd_per_mtok=25.00,
    ),
    ModelSpec(
        id="claude-opus-4-8",
        provider=PROVIDER,
        display="Claude Opus 4.8",
        context_window=1_000_000,
        max_output_tokens=128_000,
        input_usd_per_mtok=5.00,
        output_usd_per_mtok=25.00,
    ),
    ModelSpec(
        id="claude-sonnet-5",
        provider=PROVIDER,
        display="Claude Sonnet 5",
        context_window=1_000_000,
        max_output_tokens=128_000,
        input_usd_per_mtok=2.00,
        output_usd_per_mtok=10.00,
    ),
    ModelSpec(
        id="claude-sonnet-4-6",
        provider=PROVIDER,
        display="Claude Sonnet 4.6",
        context_window=1_000_000,
        max_output_tokens=128_000,
        input_usd_per_mtok=3.00,
        output_usd_per_mtok=15.00,
    ),
    ModelSpec(
        id="claude-haiku-4-5",
        provider=PROVIDER,
        display="Claude Haiku 4.5",
        context_window=200_000,
        # Best-effort: Haiku-tier output cap; not independently confirmed the way
        # the 128K figure above is for the Opus/Sonnet/Fable tier.
        max_output_tokens=64_000,
        input_usd_per_mtok=1.00,
        output_usd_per_mtok=5.00,
    ),
)

# Curated ids that still accept ``temperature`` (see the module docstring). An id
# absent from this set -- including any id we do not recognise at all -- has
# ``temperature`` omitted rather than guessed at, because the two failure modes
# are not symmetric: omitting it costs us the API's default sampling, while
# sending it to a model that removed it fails the whole call with a 400. New
# Anthropic models are overwhelmingly likely to be in the removed camp, so
# "unknown means omit" is also the way that rots most gracefully.
_ACCEPTS_SAMPLING_PARAMS: frozenset[str] = frozenset(
    {
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    }
)


def _build(model_id: str, settings: ModelSettings) -> BaseChatModel:
    def _loader() -> type:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic

    kwargs = {}
    if model_id in _ACCEPTS_SAMPLING_PARAMS:
        kwargs["temperature"] = settings.temperature

    return build_chat_model(
        provider=PROVIDER,
        env_key=ENV_KEY,
        package=PACKAGE,
        extra=EXTRA,
        loader=_loader,
        model=model_id,
        api_key=os.environ.get(ENV_KEY, ""),
        max_tokens=settings.max_tokens,
        timeout=settings.timeout_s,
        max_retries=settings.max_retries,
        streaming=settings.streaming,
        **kwargs,
    )


SPEC = ProviderSpec(
    name=PROVIDER,
    label="Anthropic",
    env_key=ENV_KEY,
    package=PACKAGE,
    extra=EXTRA,
    default_model=DEFAULT_MODEL,
    models=MODELS,
    builder=_build,
)
