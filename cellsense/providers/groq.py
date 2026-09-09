"""Groq provider: langchain_groq.ChatGroq.

DEVIATION from the original brief: ``llama-3.3-70b-versatile`` (the id given
as "known working", and what the original prototype used) is no longer served
by Groq -- a live call returns ``404 model_not_found`` and it is absent from
``GET /openai/v1/models``. Groq's hosted catalog has moved on to a different
lineup entirely (no more ``llama-3.x`` or ``gemma2`` ids). The
curated list and default below were rebuilt from the live models list and
verified with an actual tool-calling round-trip (see the provider layer's
verification report) rather than left pointing at a dead model.

Pricing below is standard per-million-token USD, sourced from
https://console.groq.com/docs/models (checked 2026-09-09). ``groq/compound`` has
no price: Groq's own model table shows a dash (no rate published) for it -- it is
an agentic/tool-orchestration system, not charged like a plain per-token model.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from cellsense.providers.base import ModelSettings, ModelSpec, ProviderSpec, build_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

PROVIDER = "groq"
ENV_KEY = "GROQ_API_KEY"
PACKAGE = "langchain_groq"
EXTRA = "groq"
DEFAULT_MODEL = "openai/gpt-oss-120b"

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="openai/gpt-oss-120b",
        provider=PROVIDER,
        display="GPT-OSS 120B (Groq)",
        # Best-effort: context/output-cap figures are typical for this open-weight
        # tier, not independently confirmed against a live capabilities endpoint
        # (unlike the pricing below, which is).
        context_window=128_000,
        max_output_tokens=32_768,
        input_usd_per_mtok=0.15,
        output_usd_per_mtok=0.60,
    ),
    ModelSpec(
        id="openai/gpt-oss-20b",
        provider=PROVIDER,
        display="GPT-OSS 20B (Groq)",
        context_window=128_000,
        max_output_tokens=32_768,
        input_usd_per_mtok=0.075,
        output_usd_per_mtok=0.30,
    ),
    ModelSpec(
        id="groq/compound",
        provider=PROVIDER,
        display="Groq Compound",
        context_window=128_000,
        max_output_tokens=8_192,
        input_usd_per_mtok=None,
        output_usd_per_mtok=None,
    ),
)


def _build(model_id: str, settings: ModelSettings) -> BaseChatModel:
    def _loader() -> type:
        from langchain_groq import ChatGroq

        return ChatGroq

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
    label="Groq",
    env_key=ENV_KEY,
    package=PACKAGE,
    extra=EXTRA,
    default_model=DEFAULT_MODEL,
    models=MODELS,
    builder=_build,
)
