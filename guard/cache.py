"""guard.cache — Redis-backed classification cache.

Architecture
------------
This module is the *only* place that knows about Redis.  The rest of the
guard package imports ``get_cache()`` and calls ``cache.get()`` /
``cache.set()`` — it never touches the Redis client directly.

Cache contract
--------------
* The cache stores **full GuardResult JSON** (not a simple "safe/dangerous"
  flag).  A cache hit returns the exact same rich object the live pipeline
  would have produced: category, threat types, confidence, rationale, all
  sub-verdicts (rewrite / validation / verification / sandbox), and the
  full audit_log.  A ``"cache=redis_hit"`` entry is appended to the
  audit_log so callers can tell the result came from cache.

* Key normalisation is deterministic:
    1. lowercase
    2. strip leading/trailing whitespace
    3. collapse interior runs of whitespace to a single space
    4. SHA-256 the result, take the full hex digest
    5. prefix with ``"apg:v1:"``

* TTL is configurable via the ``CACHE_TTL`` environment variable
  (default 86400 s = 24 h; set to 172800 for 48 h).

* Redis is optional.  If ``REDIS_URL`` is not set, ``get_cache()``
  returns ``None`` and the caller skips the cache silently.

* Every Redis error is caught, logged, and swallowed.  The guard
  pipeline is never affected by a Redis failure.

Logging
-------
The module uses the standard ``logging`` module under the logger name
``"guard.cache"``.  Expected messages:

    INFO  guard.cache — CACHE HIT  key=apg:v1:<hash>
    INFO  guard.cache — CACHE MISS key=apg:v1:<hash>
    INFO  guard.cache — stored classification in cache  key=apg:v1:<hash> ttl=86400s
    WARNING guard.cache — Redis error (get): <exception>
    WARNING guard.cache — Redis error (set): <exception>
    WARNING guard.cache — Redis unavailable, cache disabled: <exception>
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re

log = logging.getLogger("guard.cache")

# Compiled once at import time for fast normalisation.
_MULTI_SPACE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Prompt normalisation + key generation
# ---------------------------------------------------------------------------


def normalize_for_cache(prompt: str) -> str:
    """Normalise *prompt* to a canonical string for cache keying.

    Steps (must be applied in order):
    1. lowercase
    2. strip leading/trailing whitespace
    3. collapse interior whitespace runs to a single space
    """
    return _MULTI_SPACE.sub(" ", prompt.lower().strip())


# Bump when a pipeline change affects cached results' validity (e.g. a
# detection rule change) so old entries are naturally invalidated by key
# mismatch rather than needing an explicit flush.
PIPELINE_VERSION = "v2"


def make_key(prompt: str, execute: bool, version: str) -> str:
    """Return the single cache key used by *both* the Redis and in-process
    LRU caches, so a prompt screened under one config/execute mode never
    collides with — or masks — a result produced under a different one.

    Format: ``"apg:<sha256_hex>"``. The SHA-256 is computed over
    ``version:execute:normalised_prompt`` so that:
      * semantically identical prompts (same casing/spacing) share a key,
      * a prompt checked with ``execute=False`` never hits a cache entry
        produced with ``execute=True`` (or vice versa) — a cached result
        with no sandbox output must not be served for an execute request,
      * changing ``version`` (``PIPELINE_VERSION`` plus a hash of the
        active pipeline config — see ``PromptGuard.__init__``) invalidates
        every previously cached entry at once.
    """
    normalised = normalize_for_cache(prompt)
    digest = hashlib.sha256(f"{version}:{execute}:{normalised}".encode("utf-8")).hexdigest()
    return f"apg:{digest}"


# ---------------------------------------------------------------------------
# RedisCache
# ---------------------------------------------------------------------------


class RedisCache:
    """Thin wrapper around a Redis connection providing get/set for GuardResult.

    Parameters
    ----------
    url:
        Redis connection URL, e.g. ``"redis://localhost:6379/0"``.
    ttl:
        Time-to-live in seconds for every cached classification.
    """

    def __init__(self, url: str, ttl: int = 86400) -> None:
        # Import here so redis is only required when actually used.
        import redis  # type: ignore[import-untyped]

        self._ttl = ttl
        self._client = redis.Redis.from_url(url, decode_responses=True)
        # Eagerly ping to detect bad URLs / unreachable servers at construction
        # time rather than on the first cache lookup.
        self._client.ping()
        log.info("guard.cache — connected to Redis at %s (TTL=%ds)", url, ttl)

    # ── Public interface ──────────────────────────────────────────────────

    def get(self, prompt: str, execute: bool, version: str):
        """Return the cached :class:`~guard.schemas.GuardResult` for
        *prompt*/*execute*/*version*, or ``None``.

        Appends ``"cache=redis_hit"`` to the returned result's ``audit_log``
        so callers can distinguish cached results from live ones. The
        result is freshly deserialised from JSON on every call, so it is
        never the same Python object twice — safe for a caller to mutate
        without affecting what a later call returns.

        Never raises — any Redis error returns ``None`` and logs a WARNING.
        """
        # Import inside method to avoid circular import at module level.
        from .schemas import GuardResult  # noqa: PLC0415

        key = make_key(prompt, execute, version)
        try:
            raw = self._client.get(key)
        except Exception as exc:  # noqa: BLE001
            log.warning("guard.cache — Redis error (get): %s", exc)
            return None

        if raw is None:
            log.info("guard.cache — CACHE MISS  key=%s", key)
            return None

        log.info("guard.cache — CACHE HIT   key=%s", key)
        try:
            data = json.loads(raw)
            result = GuardResult.model_validate(data)
            # Append a marker so the caller / UI can surface "served from cache".
            result.audit_log.append("cache=redis_hit")
            return result
        except Exception as exc:  # noqa: BLE001
            # Corrupt or schema-incompatible cache entry — treat as a miss.
            log.warning("guard.cache — failed to deserialise cached result: %s", exc)
            return None

    def set(self, prompt: str, result, execute: bool, version: str) -> None:
        """Serialise *result* and store it in Redis with the configured TTL.

        Never raises — any Redis error is logged as a WARNING and swallowed.
        """
        key = make_key(prompt, execute, version)
        try:
            payload = result.model_dump_json()
            self._client.setex(key, self._ttl, payload)
            log.info(
                "guard.cache — stored classification in cache  key=%s ttl=%ds",
                key,
                self._ttl,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("guard.cache — Redis error (set): %s", exc)

    def ping(self) -> bool:
        """Return True if the Redis connection is alive, False otherwise."""
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False


# ---------------------------------------------------------------------------
# Module-level singleton — created once per process from env vars
# ---------------------------------------------------------------------------

_cache_instance: "RedisCache | None | _Sentinel" = None


class _Sentinel:
    """Marks that get_cache() was already called and returned None."""


_SENTINEL = _Sentinel()


def get_cache() -> "RedisCache | None":
    """Return the module-level :class:`RedisCache` singleton, or ``None``.

    On first call, reads ``REDIS_URL`` and ``CACHE_TTL`` from the environment.
    If ``REDIS_URL`` is not set, or if Redis is unreachable, returns ``None``
    and caches that decision so subsequent calls are instant.

    Subsequent calls return the same (possibly ``None``) singleton without
    re-reading the environment or re-connecting.
    """
    global _cache_instance  # noqa: PLW0603

    if isinstance(_cache_instance, _Sentinel):
        return None
    if _cache_instance is not None:
        return _cache_instance

    redis_url = os.environ.get("REDIS_URL", "").strip()
    if not redis_url:
        log.debug("guard.cache — REDIS_URL not set; cache disabled")
        _cache_instance = _SENTINEL
        return None

    ttl_raw = os.environ.get("CACHE_TTL", "86400").strip()
    try:
        ttl = int(ttl_raw)
        if ttl <= 0:
            raise ValueError("TTL must be positive")
    except ValueError:
        log.warning(
            "guard.cache — invalid CACHE_TTL=%r; using default 86400s", ttl_raw
        )
        ttl = 86400

    try:
        _cache_instance = RedisCache(url=redis_url, ttl=ttl)
        return _cache_instance
    except Exception as exc:  # noqa: BLE001
        log.warning("guard.cache — Redis unavailable, cache disabled: %s", exc)
        _cache_instance = _SENTINEL
        return None
