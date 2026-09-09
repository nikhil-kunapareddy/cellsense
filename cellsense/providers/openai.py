"""OpenAI provider: langchain_openai.ChatOpenAI.

CellSense's own .env only ever carries GROQ/GEMINI/LLAMA keys (see CLAUDE.md); this
provider can be *built* against a supplied OPENAI_API_KEY but is not part of the
live-call verification matrix for this change. Pricing below is standard
(uncached, non-batch, text) per-million-token USD, sourced from
https://developers.openai.com/api/docs/pricing (checked 2026-09-09; this is the
current URL platform.openai.com/docs/pricing 301-redirects to).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from cellsense.providers.base import ModelSettings, ModelSpec, ProviderSpec, build_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

PROVIDER = "openai"
ENV_KEY = "OPENAI_API_KEY"
PACKAGE = "langchain_openai"
EXTRA = "openai"
DEFAULT_MODEL = "gpt-4o"

MODELS: tuple[ModelSpec, ...] = (
    # Pricing source: https://developers.openai.com/api/docs/pricing (checked 2026-09-09).
    ModelSpec(
        id="gpt-4o",
        provider=PROVIDER,
        display="GPT-4o",
        context_window=128_000,
        max_output_tokens=16_384,
        input_usd_per_mtok=2.50,
        output_usd_per_mtok=10.00,
    ),
    ModelSpec(
        id="gpt-4o-mini",
        provider=PROVIDER,
        display="GPT-4o mini",
        context_window=128_000,
        max_output_tokens=16_384,
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.60,
    ),
    ModelSpec(
        id="gpt-4.1",
        provider=PROVIDER,
        display="GPT-4.1",
        context_window=1_047_576,
        max_output_tokens=32_768,
        input_usd_per_mtok=2.00,
        output_usd_per_mtok=8.00,
    ),
    ModelSpec(
        id="gpt-4.1-mini",
        provider=PROVIDER,
        display="GPT-4.1 mini",
        context_window=1_047_576,
        max_output_tokens=32_768,
        input_usd_per_mtok=0.40,
        output_usd_per_mtok=1.60,
    ),
)


def _build(model_id: str, settings: ModelSettings) -> BaseChatModel:
    def _loader() -> type:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI

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
    label="OpenAI",
    env_key=ENV_KEY,
    package=PACKAGE,
    extra=EXTRA,
    default_model=DEFAULT_MODEL,
    models=MODELS,
    builder=_build,
)
