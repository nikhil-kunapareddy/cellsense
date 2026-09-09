"""Provider-agnostic building blocks: specs, settings, error translation, and the
lazy chat-model construction helper every provider module shares.

Nothing in this module imports a ``langchain_*`` integration package. Provider
modules (``anthropic.py``, ``openai.py``, ``gemini.py``, ``groq.py``, ``llama.py``)
each own exactly one such import, made lazily inside their ``build()`` closure, so
that ``import cellsense.providers.anthropic`` never pulls in the Anthropic SDK.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cellsense.errors import (
    AuthenticationError,
    CellSenseError,
    MissingAPIKeyError,
    MissingDependencyError,
    ProviderError,
    RateLimitError,
)

if TYPE_CHECKING:
    # Only for type checkers; never imported at runtime by this module.
    from langchain_core.language_models.chat_models import BaseChatModel

__all__ = [
    "ModelSettings",
    "ModelSpec",
    "ProviderSpec",
    "build_chat_model",
    "translate_provider_error",
]


@dataclass(frozen=True)
class ModelSettings:
    """Per-call knobs, sourced from :class:`cellsense.config.Config` and passed
    straight through to the LangChain constructor -- CellSense never hand-rolls a
    retry loop or a timeout clock (see ARCHITECTURE.md §4 rule 5).
    """

    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: int = 120
    max_retries: int = 3
    streaming: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for one concrete model id.

    ``input_usd_per_mtok`` / ``output_usd_per_mtok`` are ``None`` when pricing is
    unknown -- either because the id was synthesized for an unrecognized selector
    (see ``registry.resolve_model``), or because we could not confirm a real number
    and refused to invent one. ``None`` pricing means cost is displayed as ``"-"``,
    never ``$0.00``.
    """

    id: str
    provider: str
    display: str
    context_window: int
    max_output_tokens: int
    input_usd_per_mtok: float | None
    output_usd_per_mtok: float | None
    supports_tools: bool = True
    supports_streaming: bool = True


@dataclass(frozen=True)
class ProviderSpec:
    """Everything needed to describe and build models for one backend.

    ``builder`` is the provider module's private ``_build`` function -- the only
    place that actually imports the ``langchain_*`` package, invoked lazily by
    :meth:`build`. Keeping it as a plain callable field (rather than a method
    overridden per-module) lets ``ProviderSpec`` stay a single frozen dataclass
    shape that ``registry.py`` can treat uniformly.
    """

    name: str
    label: str
    env_key: str
    package: str
    extra: str
    default_model: str
    models: tuple[ModelSpec, ...]
    builder: Callable[[str, ModelSettings], BaseChatModel]

    def build(self, model_id: str, settings: ModelSettings) -> BaseChatModel:
        """Construct a bound chat model for ``model_id``. Raises
        :class:`MissingAPIKeyError`, :class:`MissingDependencyError`, or a
        translated :class:`ProviderError` subclass -- never a raw SDK exception.
        """
        return self.builder(model_id, settings)


def translate_provider_error(exc: Exception, provider: str) -> CellSenseError:
    """Map an arbitrary provider SDK exception onto the CellSense error hierarchy.

    This inspects status codes and class names generically -- it never imports a
    provider SDK to do so (that would defeat the point of lazy imports). Every
    SDK this project talks to (anthropic, openai, google, groq) surfaces an HTTP
    status somewhere on the exception, either as ``.status_code``, ``.status``, or
    nested under ``.response.status_code``; when none is present we fall back to
    sniffing the exception class name.
    """
    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "status", None)
        or getattr(getattr(exc, "response", None), "status_code", None)
    )
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None

    cls_name = type(exc).__name__.lower()
    message = f"{provider}: {exc}"

    if status in (401, 403) or "authenticat" in cls_name or "permission" in cls_name:
        return AuthenticationError(message)
    if status == 429 or "ratelimit" in cls_name or "rate_limit" in cls_name or "quota" in cls_name:
        return RateLimitError(message)
    return ProviderError(message)


def build_chat_model(
    *,
    provider: str,
    env_key: str,
    package: str,
    extra: str,
    loader: Callable[[], type],
    **kwargs: Any,
) -> BaseChatModel:
    """Shared lazy-import + construct + error-translate path for every provider.

    Checks for the API key first (so a missing key is reported as
    :class:`MissingAPIKeyError` before we even try to import anything), then calls
    ``loader()`` -- a zero-arg closure the provider module defines to do
    ``from langchain_x import ChatX; return ChatX`` -- converting ``ImportError``
    into :class:`MissingDependencyError`. Construction failures (bad key format,
    network probes some SDKs do at init) are translated via
    :func:`translate_provider_error`.
    """
    if not os.environ.get(env_key):
        raise MissingAPIKeyError(provider, env_key)

    try:
        model_cls = loader()
    except ImportError as exc:
        raise MissingDependencyError(provider, package, extra) from exc

    try:
        return model_cls(**kwargs)
    except CellSenseError:
        raise
    except Exception as exc:
        raise translate_provider_error(exc, provider) from exc
