"""Anthropic provider: langchain_anthropic.ChatAnthropic.

Pricing and context/output limits below are the Claude 5 generation figures current
as of this writing. ``claude-sonnet-5`` is the provider default -- it is the
balanced-cost model in the family, matching the role ``claude-sonnet-4-5`` played
in the original prototype's Anthropic backend this module supersedes.
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
DEFAULT_MODEL = "claude-sonnet-5"

MODELS: tuple[ModelSpec, ...] = (
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
        id="claude-sonnet-5",
        provider=PROVIDER,
        display="Claude Sonnet 5",
        context_window=1_000_000,
        max_output_tokens=128_000,
        input_usd_per_mtok=2.00,
        output_usd_per_mtok=10.00,
    ),
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
        id="claude-haiku-4-5-20251001",
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


def _build(model_id: str, settings: ModelSettings) -> BaseChatModel:
    def _loader() -> type:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic

    return build_chat_model(
        provider=PROVIDER,
        env_key=ENV_KEY,
        package=PACKAGE,
        extra=EXTRA,
        loader=_loader,
        model=model_id,
        api_key=os.environ.get(ENV_KEY, ""),
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout_s,
        max_retries=settings.max_retries,
        streaming=settings.streaming,
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
