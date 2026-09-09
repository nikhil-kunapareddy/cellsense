"""Bring-your-own OpenAI-compatible Llama host.

Meta retired its first-party hosted Llama API (``api.llama.com``) on 2026-07-06
(https://llama.developer.meta.com/docs/llama-api-deprecation/) -- this is
confirmed, not a guess: every model id this module previously tried returned
``400 Invalid model name``, ``GET /v1/models`` returned an empty list, and a
deliberately wrong key still got a distinct ``401`` (proving the key itself
authenticated fine against a service with zero models behind it). There is
nothing left to point this provider at by default.

This provider therefore no longer targets Meta at all. It is a thin
"llama"-branded front door onto **whatever OpenAI-compatible endpoint the user
supplies** via ``LLAMA_BASE_URL`` -- Together AI, AWS Bedrock, Fireworks, a
self-hosted vLLM/TGI server, or any other host that speaks the OpenAI chat
completions wire format. ``langchain_openai.ChatOpenAI`` is still the right
client for that (same approach the legacy
the original prototype's LLAMA backend config took, minus the hardcoded
Meta endpoint).

``LLAMA_BASE_URL`` is **required** -- there is deliberately no default and no
fallback to the retired ``api.llama.com``. If it is unset, :func:`_build` raises
:class:`ConfigError` before importing anything or touching the network (a
doomed HTTP request is worse than a clear error up front).

Model ids are **host-specific** and the curated list below is illustrative
only, not authoritative: Together AI names this model
``meta-llama/Llama-3.3-70B-Instruct-Turbo``, Bedrock addresses it by ARN, a
self-hosted vLLM server might expose it under whatever name it was launched
with, etc. ``Llama-3.3-70B-Instruct`` is kept as the nominal default id (it is
the plain upstream Llama name most self-hosted servers use verbatim), but
callers on Together/Bedrock/etc. are expected to pass their host's actual id via
``--model llama:<host-specific-id>`` -- ``registry.resolve_model``'s synthesis
path already accepts any id here without whitelisting it (ARCHITECTURE.md §4
rule 2), so no change was needed there.

Pricing is left as ``None`` for every curated id, and that is now the correct,
permanent answer rather than a gap to fill in later: cost depends entirely on
which host ``LLAMA_BASE_URL`` points at, and this module has no way to know
that in advance.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from cellsense.errors import ConfigError
from cellsense.providers.base import ModelSettings, ModelSpec, ProviderSpec, build_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

PROVIDER = "llama"
ENV_KEY = "LLAMA_API_KEY"
PACKAGE = "langchain_openai"
EXTRA = "llama"
DEFAULT_MODEL = "Llama-3.3-70B-Instruct"


def _require_base_url() -> str:
    """Read ``LLAMA_BASE_URL``, or raise -- there is no default host anymore.

    Called from within :func:`_build`'s keyword arguments, so this executes (and
    can raise) before ``build_chat_model`` does its own lazy import or API-key
    check, and long before any request is sent.
    """
    base_url = os.environ.get("LLAMA_BASE_URL")
    if not base_url:
        raise ConfigError(
            "Provider 'llama' needs LLAMA_BASE_URL.",
            hint=(
                "Meta's hosted Llama API (api.llama.com) was retired on "
                "2026-07-06. Point LLAMA_BASE_URL at any OpenAI-compatible Llama "
                "host (Together AI, AWS Bedrock, Fireworks, self-hosted vLLM), "
                "or use --model groq:... instead."
            ),
        )
    return base_url


# Illustrative only -- see the module docstring. Real ids are whatever the host
# behind LLAMA_BASE_URL calls them; unrecognized ids synthesize cleanly via
# registry.resolve_model rather than being rejected.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="Llama-3.3-70B-Instruct",
        provider=PROVIDER,
        display="Llama 3.3 70B Instruct",
        context_window=128_000,
        max_output_tokens=8_192,
        # Depends on the chosen host -- not a gap, there is no single answer.
        input_usd_per_mtok=None,
        output_usd_per_mtok=None,
    ),
    ModelSpec(
        id="Llama-4-Maverick-17B-128E-Instruct-FP8",
        provider=PROVIDER,
        display="Llama 4 Maverick 17Bx128E Instruct FP8",
        context_window=1_000_000,
        max_output_tokens=8_192,
        input_usd_per_mtok=None,
        output_usd_per_mtok=None,
    ),
    ModelSpec(
        id="Llama-4-Scout-17B-16E-Instruct-FP8",
        provider=PROVIDER,
        display="Llama 4 Scout 17Bx16E Instruct FP8",
        context_window=10_000_000,
        max_output_tokens=8_192,
        input_usd_per_mtok=None,
        output_usd_per_mtok=None,
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
        base_url=_require_base_url(),
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout_s,
        max_retries=settings.max_retries,
        streaming=settings.streaming,
    )


SPEC = ProviderSpec(
    name=PROVIDER,
    label="Llama (self-hosted / third-party)",
    env_key=ENV_KEY,
    package=PACKAGE,
    extra=EXTRA,
    default_model=DEFAULT_MODEL,
    models=MODELS,
    builder=_build,
)
