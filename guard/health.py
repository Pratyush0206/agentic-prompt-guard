"""Layer health reporting and fail-closed mode (P2-7).

Two optional layers can silently degrade to a weaker fallback: the Ollama
LLM (falls back to deterministic heuristics) and the sentence-transformers
semantic matcher (falls back to a difflib string-ratio comparison). By
default the guard logs this once and keeps running — a deployment that
would rather refuse requests than run degraded passes
``PromptGuard(strict=True)`` instead, which raises :class:`LayerUnavailable`
at construction time.

The Redis cache is intentionally excluded from the strict-mode check: it is
optional-by-design (most deployments never set ``REDIS_URL``), so its
absence is not "degraded" the way a requested-but-missing LLM or semantic
matcher is. It still appears in :func:`layer_status` for visibility.
"""

from __future__ import annotations

import logging

log = logging.getLogger("guard.health")


class LayerUnavailable(RuntimeError):
    """Raised by ``PromptGuard(strict=True)`` when a requested layer (the
    LLM or the semantic matcher) is not actually available, instead of
    silently falling back to a weaker deterministic path."""


def layer_status(
    *,
    llm_requested: bool,
    llm_active: bool,
    model: str,
    semantic_active: bool,
    redis_active: bool,
) -> dict[str, str]:
    """Return a snapshot of which optional layers are active vs. degraded.

    Keys are ``"llm"``, ``"semantic"``, ``"redis"``; values are short,
    machine-readable status strings suitable for logging or an audit_log
    entry (see :func:`format_layer_status`).
    """
    if llm_requested:
        llm = f"active:{model}" if llm_active else "inactive (requested but unreachable)"
    else:
        llm = "not_requested (heuristic-only)"
    semantic = "sentence-transformers" if semantic_active else "difflib_fallback"
    redis = "active" if redis_active else "disabled"
    return {"llm": llm, "semantic": semantic, "redis": redis}


def format_layer_status(status: dict[str, str]) -> str:
    """Render a :func:`layer_status` dict as a single audit_log line."""
    return "layer_status: " + " ".join(f"{k}={v}" for k, v in status.items())


def is_degraded(*, llm_requested: bool, llm_active: bool, semantic_active: bool) -> bool:
    """True when a layer the pipeline actually wanted isn't available."""
    return (llm_requested and not llm_active) or not semantic_active


def check_strict(*, strict: bool, llm_requested: bool, llm_active: bool, semantic_active: bool) -> None:
    """Raise :class:`LayerUnavailable` when ``strict`` and a requested
    layer (LLM or semantic matcher) is missing. No-op when not strict."""
    if not strict:
        return
    if llm_requested and not llm_active:
        raise LayerUnavailable(
            "LLM backend requested (use_llm=True) but Ollama is not reachable; "
            "refusing to silently fall back to heuristics under strict=True."
        )
    if not semantic_active:
        raise LayerUnavailable(
            "Semantic matcher requested but sentence-transformers is not installed "
            "or failed to load; refusing to silently fall back to difflib under strict=True."
        )
