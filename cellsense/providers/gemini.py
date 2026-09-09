"""Google Gemini provider: langchain_google_genai.ChatGoogleGenerativeAI.

DEVIATION from the original brief: the brief's default, ``gemini-2.0-flash``
(what the original prototype used), is
retired as of this writing -- a live call returns ``404 NOT_FOUND`` with the
API's own message pointing callers at a newer id. ``gemini-2.5-flash`` is used
as the default instead: it is still present in ``GET /v1beta/models`` against
the live GEMINI_API_KEY in this repo's .env and was verified with an actual
tool-calling round-trip. ``gemini-2.5-pro`` and ``gemini-2.0-flash`` remain in
the curated list per the original brief (the latter for continuity/reference),
but selecting ``gemini-2.0-flash`` will itself now fail live -- the curated list
is metadata, not a guarantee of availability.

Pricing below is standard paid-tier, uncached, non-batch, text per-million-token
USD, sourced from https://ai.google.dev/gemini-api/docs/pricing (checked
2026-09-09). ``gemini-2.5-pro``'s rate is the <=200K-token-prompt tier (it rises
to $2.50/$15.00 above 200K input tokens -- not modeled here, since ``ModelSpec``
has no tiering concept yet). ``gemini-2.0-flash`` has no price: it is absent from
the live pricing page entirely, consistent with the 404 in the DEVIATION note
above -- this is a genuinely retired model, not an unpublished price.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from cellsense.providers.base import ModelSettings, ModelSpec, ProviderSpec, build_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

PROVIDER = "gemini"
ENV_KEY = "GEMINI_API_KEY"
PACKAGE = "langchain_google_genai"
EXTRA = "gemini"
DEFAULT_MODEL = "gemini-2.5-flash"

MODELS: tuple[ModelSpec, ...] = (
    # Pricing source: https://ai.google.dev/gemini-api/docs/pricing (checked 2026-09-09).
    ModelSpec(
        id="gemini-2.5-pro",
        provider=PROVIDER,
        display="Gemini 2.5 Pro",
        context_window=1_048_576,
        max_output_tokens=65_536,
        input_usd_per_mtok=1.25,
        output_usd_per_mtok=10.00,
    ),
    ModelSpec(
        id="gemini-2.5-flash",
        provider=PROVIDER,
        display="Gemini 2.5 Flash",
        context_window=1_048_576,
        max_output_tokens=65_536,
        input_usd_per_mtok=0.30,
        output_usd_per_mtok=2.50,
    ),
    ModelSpec(
        id="gemini-2.0-flash",
        provider=PROVIDER,
        display="Gemini 2.0 Flash",
        context_window=1_048_576,
        max_output_tokens=8_192,
        # Not None as an oversight: this model is genuinely absent from the live
        # pricing page (retired), so there is no published price to record.
        input_usd_per_mtok=None,
        output_usd_per_mtok=None,
    ),
)


def _build(model_id: str, settings: ModelSettings) -> BaseChatModel:
    def _loader() -> type:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI

    return build_chat_model(
        provider=PROVIDER,
        env_key=ENV_KEY,
        package=PACKAGE,
        extra=EXTRA,
        loader=_loader,
        model=model_id,
        api_key=os.environ.get(ENV_KEY, ""),
        temperature=settings.temperature,
        max_output_tokens=settings.max_tokens,
        timeout=settings.timeout_s,
        max_retries=settings.max_retries,
        streaming=settings.streaming,
    )


SPEC = ProviderSpec(
    name=PROVIDER,
    label="Google Gemini",
    env_key=ENV_KEY,
    package=PACKAGE,
    extra=EXTRA,
    default_model=DEFAULT_MODEL,
    models=MODELS,
    builder=_build,
)
