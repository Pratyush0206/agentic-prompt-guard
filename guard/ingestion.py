"""Step 1 — Intent & Context Ingestion.

Pre-processes the raw prompt before any LLM call or regex scan.  Four passes
are applied in sequence:

1. Invisible-character stripping + NFKC normalisation (zero-width, bidi chars).
2. Homoglyph substitution — Cyrillic/Greek look-alikes replaced with ASCII.
3. Whitespace-injection defuser — collapses 'i g n o r e' into 'ignore'.
4. Leetspeak normalisation — '1gnore @ll' becomes 'ignore all'.

Encoded payloads (base64 / hex / ROT13) are decoded *before* the leetspeak
pass to prevent digit substitutions from corrupting valid tokens.  The decoded
strings are added to the regex scan haystacks.

Semantic similarity is computed via :mod:`guard.semantic` (sentence-transformers
when available, difflib ratio as fallback).
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Normalisation tables
# ---------------------------------------------------------------------------

# Zero-width and bidirectional control characters used to hide payloads.
_INVISIBLE = dict.fromkeys(
    map(
        ord,
        (
            "\u200b"  # zero width space
            "\u200c"  # zero width non-joiner
            "\u200d"  # zero width joiner
            "\u2060"  # word joiner
            "\ufeff"  # BOM / zero width no-break space
            "\u202a"  # left-to-right embedding
            "\u202b"  # right-to-left embedding
            "\u202c"  # pop directional formatting
            "\u202d"  # left-to-right override
            "\u202e"  # right-to-left override
        ),
    ),
    None,
)

# Homoglyph table: maps ord() of a confusable non-ASCII character to its
# ASCII string equivalent.  Covers the most common Cyrillic and Greek
# confusables used in bypass attacks.
_HOMOGLYPH: dict[int, str] = {
    # ── Cyrillic confusables ──────────────────────────────────────────────
    ord("а"): "a",  ord("А"): "A",   # U+0430 / U+0410
    ord("е"): "e",  ord("Е"): "E",   # U+0435 / U+0415
    ord("о"): "o",  ord("О"): "O",   # U+043E / U+041E
    ord("р"): "p",  ord("Р"): "P",   # U+0440 / U+0420
    ord("с"): "c",  ord("С"): "C",   # U+0441 / U+0421
    ord("х"): "x",  ord("Х"): "X",   # U+0445 / U+0425
    ord("у"): "y",  ord("У"): "Y",   # U+0443 / U+0423
    ord("і"): "i",  ord("І"): "I",   # U+0456 / U+0406
    ord("ѕ"): "s",                    # U+0455
    ord("ј"): "j",                    # U+0458
    ord("В"): "B",                    # U+0412
    ord("К"): "K",                    # U+041A
    ord("М"): "M",                    # U+041C
    ord("Н"): "H",                    # U+041D
    ord("Т"): "T",                    # U+0422
    # ── Greek confusables ────────────────────────────────────────────────
    ord("α"): "a",  ord("Α"): "A",
    ord("β"): "b",  ord("Β"): "B",
    ord("ε"): "e",  ord("Ε"): "E",
    ord("η"): "n",  ord("Η"): "H",
    ord("ι"): "i",  ord("Ι"): "I",
    ord("κ"): "k",  ord("Κ"): "K",
    ord("μ"): "u",  ord("Μ"): "M",
    ord("ν"): "v",  ord("Ν"): "N",
    ord("ο"): "o",  ord("Ο"): "O",
    ord("ρ"): "p",  ord("Ρ"): "P",
    ord("τ"): "t",  ord("Τ"): "T",
    ord("υ"): "u",  ord("Υ"): "Y",
    ord("χ"): "x",  ord("Χ"): "X",
    ord("ω"): "w",  ord("Ω"): "W",
}

# Leetspeak substitution map — single characters only (applied char-by-char).
# Note: this is aggressive by design; it may alter legitimate numeric text
# (e.g. "Q1" → "Qi"). Because of that, the leet-normalized text is used only
# as one of several scan variants (see IngestionResult.variants) — never as
# the primary `normalized` text, which must keep ordinary numbers intact
# (P2-3). The original prompt is preserved in IngestionResult.original.
_LEET: dict[str, str] = {
    "1": "i",
    "!": "i",
    "|": "i",
    "0": "o",
    "3": "e",
    "@": "a",
    "4": "a",
    "$": "s",
    "5": "s",
    "7": "t",
    "+": "t",
    "8": "b",
    "6": "g",
    "9": "g",
}

# ---------------------------------------------------------------------------
# Jailbreak signatures (expanded)
# ---------------------------------------------------------------------------

_JAILBREAK_SIGNATURES: list[str] = [
    # ── Classic prompt-injection phrases ──────────────────────────────────
    r"ignore (all )?(previous|prior|above|your) (instructions|prompts|rules|directives|training|restrictions)",
    r"disregard (the )?(system|previous|prior|all) (prompt|instructions|guidelines|rules|training)",
    r"do anything now",
    r"developer mode",
    r"pretend (you are|to be)",
    r"act as (an? )?(unfiltered|jailbroken|uncensored|unrestricted)",
    r"you have no (restrictions|rules|guidelines|limits|filters)",
    r"bypass (your )?(safety|content|guardrail|filter|restriction)",
    r"reveal (your )?(system prompt|instructions|training|guidelines)",
    # Chat template / tag injections
    r"\[\[\s*(SYSTEM|ADMIN|DEVELOPER|PROMPT)\s*\]\]",
    r"\[/?INST\]",
    r"<\|im_(start|end)\|>",
    # ── Forget / override instructions ───────────────────────────────────
    r"(forget|disregard|ignore).{0,20}(everything|what|all).{0,20}"
    r"(told|said|instructed|programmed|trained)",
    r"(act|behave|respond).{0,20}as if.{0,30}(no|without|bypass).{0,20}"
    r"(restriction|limit|filter|rule)",
    r"(primary directive|new directive).{0,40}(ignore|disregard|override|bypass)",
    r"remember for all future.{0,40}(ignore|disregard|bypass)",
    # ── Hypothetical / roleplay bypass ───────────────────────────────────
    r"(hypothetically|imagine|suppose|what if|let.?s say|pretend).{0,80}"
    r"(fda|regulation|compliance|safety|rules|law|restriction).{0,50}"
    r"(was not|not required|didn.?t exist|do(es)? not exist|gone|ignored|removed|bypass|without|no )",
    r"(hypothetically|imagine|suppose|what if|let.?s say|pretend).{0,80}"
    r"(no|without|ignoring|bypass|not following|waiv\w*).{0,50}"
    r"(fda|regulation|compliance|safety|rules|restriction)",
    # ── Multi-step / split injection ─────────────────────────────────────
    r"(step 1|first).{0,50}(step 2|second|then).{0,50}(step 3|third|finally)",
    r"you (must|will|shall|are required to) (ignore|bypass|disregard|override)",
    # ── Encoding tricks ───────────────────────────────────────────────────
    r"encode.{0,30}(harmful|restricted|blocked|unsafe).{0,30}(base64|hex|rot)",
]

# Signatures that must stay case-SENSITIVE: "DAN" (the jailbreak persona) is
# an all-caps acronym that collides with the common first name "Dan" when
# matched case-insensitively (P1-4 regression — "My colleague Dan asked...").
# Kept separate from _JAILBREAK_SIGNATURES, which is intentionally
# case-insensitive for every other signature.
_JAILBREAK_SIGNATURES_CASE_SENSITIVE: list[str] = [
    r"\bDAN\b",
]

# Similarity threshold used for the difflib fallback path.
# The semantic matcher uses its own threshold defined in guard/semantic.py.
_SIMILARITY_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# Normalisation functions
# ---------------------------------------------------------------------------


def _strip_invisibles(text: str) -> str:
    """Remove zero-width / bidi control chars, apply NFKC normalisation, and strip chat markup."""
    t = unicodedata.normalize("NFKC", text.translate(_INVISIBLE))
    # Strip common LLM chat delimiters (e.g. </s>, [INST], [/INST], [[SYSTEM]], <|im_start|>)
    t = re.sub(r'</?[a-zA-Z0-9_|-]+>|\[/?INST\]|\[\[SYSTEM\]\]|<\|im_\w+\|>', ' ', t)
    return t


def _apply_homoglyph(text: str) -> str:
    """Replace visually-identical non-ASCII characters with ASCII equivalents."""
    return text.translate(_HOMOGLYPH)


def _defuse_whitespace_injection(text: str) -> str:
    """Collapse whitespace-injected strings like 'i g n o r e' or 'i.g.n.o.r.e' into 'ignore'.

    Detects runs of **3 or more** consecutive single-character tokens separated
    by spaces, dots, or dashes, and joins them into one word. Normal prose (words of
    length > 1) is left entirely unchanged.
    """
    # Pre-process: convert dots, dashes, or underscores between single letters into spaces
    cleaned = re.sub(r'(?<=\b[a-zA-Z0-9])[\.\-_]+(?=[a-zA-Z0-9]\b)', ' ', text)
    # Convert underscores in snake_case attack patterns to spaces (e.g. ignore_all_previous_instructions -> ignore all previous instructions)
    cleaned = re.sub(r'_+', ' ', cleaned)

    tokens = cleaned.split(" ")
    result: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # Single-char token run (e.g., "i g n o r e")
        if (
            len(tok) == 1
            and tok.isalnum()
            and i + 2 < len(tokens)
            and len(tokens[i + 1]) == 1
            and tokens[i + 1].isalnum()
        ):
            run = [tok]
            j = i + 1
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalnum():
                run.append(tokens[j])
                j += 1
            if len(run) >= 3:
                result.append("".join(run))
            else:
                result.extend(run)
            i = j
        else:
            result.append(tok)
            i += 1

    return " ".join(result)


def _apply_leetspeak(text: str) -> str:
    """Translate common leet-speak digit/symbol substitutions back to letters."""
    return "".join(_LEET.get(ch, ch) for ch in text)


# ---------------------------------------------------------------------------
# Language detection (P1-5) — stdlib-only, English vs. everything else
# ---------------------------------------------------------------------------

# Common English function words. A real English sentence of non-trivial
# length almost always contains at least one of these; a non-English
# sentence (even one written in the Latin alphabet, e.g. Spanish/French,
# which a script-ratio check alone would miss since they share most of the
# Latin alphabet with English) essentially never does.
_ENGLISH_STOPWORDS: frozenset = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "and", "or", "but", "not",
    "no", "yes", "this", "that", "these", "those", "it", "its", "you",
    "your", "i", "we", "our", "they", "their", "he", "she", "his", "her",
    "what", "which", "who", "whom", "how", "why", "when", "where", "do",
    "does", "did", "can", "could", "should", "must", "will", "would",
    "shall", "may", "might", "please", "me", "my", "us", "as", "by",
    "from", "into", "about", "if", "then", "than", "so", "such", "also",
    "all", "any", "each", "other", "some", "more", "most", "very",
})

# Below this many word tokens there isn't enough signal to judge language
# reliably, so short prompts are never flagged as unsupported-language.
_MIN_TOKENS_FOR_LANGUAGE_CHECK = 5

# Fraction of alphabetic characters that must be non-ASCII before a prompt
# is considered to be in a non-Latin script (Cyrillic, Greek, CJK, Arabic...).
_NON_ASCII_SCRIPT_THRESHOLD = 0.25


def detect_script_ratio(text: str) -> float:
    """Return the fraction of alphabetic characters in *text* that are non-ASCII."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    non_ascii = sum(1 for ch in letters if ord(ch) > 127)
    return non_ascii / len(letters)


def is_unsupported_language(text: str) -> bool:
    """Heuristic: True when *text* shows no reliable sign of being English.

    Two stdlib-only signals, combined conservatively so ordinary short or
    jargon-heavy English prompts are never misflagged:
      1. Non-ASCII letter ratio — catches non-Latin scripts outright.
      2. Zero English-stopword coverage over enough tokens — catches
         Latin-script non-English languages that signal 1 would miss.
    """
    tokens = re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE)
    if len(tokens) < _MIN_TOKENS_FOR_LANGUAGE_CHECK:
        return False
    if detect_script_ratio(text) >= _NON_ASCII_SCRIPT_THRESHOLD:
        return True
    return not any(tok in _ENGLISH_STOPWORDS for tok in tokens)


def _full_normalize(text: str) -> str:
    """Apply all four normalisation passes in sequence."""
    text = _strip_invisibles(text)
    text = _apply_homoglyph(text)
    text = _defuse_whitespace_injection(text)
    text = _apply_leetspeak(text)
    return text.strip()


# ---------------------------------------------------------------------------
# Encoded-payload decoder
# ---------------------------------------------------------------------------


# P2-4: bounds on decode work per prompt, so an adversarially large or
# repetitive input (e.g. a 100k-char string) cannot blow up decode time —
# every pass below operates on text already truncated to _MAX_CANDIDATE_LEN,
# and the total candidate count across all passes/depths is capped too.
_MAX_DECODE_CANDIDATES = 20
_MAX_CANDIDATE_LEN = 10_000
_MAX_DECODE_DEPTH = 3

_MEANINGFUL_RE = re.compile(r"[a-zA-Z]{3,}")
_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")

# Gate for speculative transforms (ROT13, string-reversal) that have no
# other signal of their own: only trust the transformed text if it surfaces
# one of these recognizable attack phrases, not merely "looks like text".
_CIPHER_JAILBREAK_RE = re.compile(r"ignore .*instructions|developer mode", re.IGNORECASE)


def _looks_meaningful(candidate: str) -> bool:
    return candidate.isprintable() and bool(_MEANINGFUL_RE.search(candidate))


def _decode_base64_tokens(text: str) -> list[str]:
    out: list[str] = []
    # Shorter minimum (12, was 16) so shorter smuggled payloads are still caught.
    for token in re.findall(r"[A-Za-z0-9+/=]{12,}", text):
        try:
            candidate = base64.b64decode(token, validate=True).decode("utf-8", "strict")
            if _looks_meaningful(candidate):
                out.append(candidate)
        except (binascii.Error, ValueError, UnicodeDecodeError):
            pass
    return out


def _decode_hex_tokens(text: str) -> list[str]:
    out: list[str] = []
    for token in re.findall(r"(?:[0-9a-fA-F]{2}\s*){8,}", text):
        try:
            raw = bytes.fromhex(re.sub(r"\s+", "", token))
            candidate = raw.decode("utf-8", "strict")
            if _looks_meaningful(candidate):
                out.append(candidate)
        except (ValueError, UnicodeDecodeError):
            pass
    return out


def _decode_urlencoded(text: str) -> list[str]:
    if "%" not in text:
        return []
    try:
        candidate = urllib.parse.unquote(text, errors="strict")
    except Exception:  # noqa: BLE001 — malformed %XX sequences, etc.
        return []
    return [candidate] if candidate != text and _looks_meaningful(candidate) else []


def _decode_unicode_escapes(text: str) -> list[str]:
    """Decode literal textual ``\\u00XX`` escape sequences (not real Unicode
    chars — those are already handled by NFKC normalisation — but the
    six-character *text* "\\u0069" etc., a common filter-evasion trick)."""
    if not _UNICODE_ESCAPE_RE.search(text):
        return []
    try:
        candidate = _UNICODE_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), text)
    except ValueError:
        return []
    return [candidate] if candidate != text and _looks_meaningful(candidate) else []


def _decode_reversed(text: str) -> list[str]:
    """Flag a reversed string as a candidate, but only when it actually
    surfaces a recognizable attack phrase (same gate as the ROT13 check
    below) — `_looks_meaningful` alone is far too weak here: reversing any
    ordinary English sentence still "looks like text" (printable, has
    letter runs), so without this gate almost every prompt's reversed form
    would be flagged as a decoded payload."""
    reversed_text = text[::-1]
    if reversed_text == text:
        return []
    if _CIPHER_JAILBREAK_RE.search(reversed_text):
        return [reversed_text]
    return []


def _decode_one_layer(text: str) -> list[str]:
    """One pass of every *recursive* decode strategy (not the reverse-string
    or ROT13 checks, which only make sense on the original text once)."""
    out: list[str] = []
    out.extend(_decode_base64_tokens(text))
    out.extend(_decode_hex_tokens(text))
    out.extend(_decode_urlencoded(text))
    out.extend(_decode_unicode_escapes(text))
    return out


def _try_decode(text: str) -> list[str]:
    """Surface hidden instructions smuggled via base64 / hex / URL-encoding /
    textual unicode escapes / ROT13 / string reversal, including nested
    encodings up to :data:`_MAX_DECODE_DEPTH` layers deep (e.g. hex-of-base64,
    or base64-of-base64). Bounded by :data:`_MAX_DECODE_CANDIDATES` total
    results and :data:`_MAX_CANDIDATE_LEN` chars per candidate (P2-4).
    """
    text = text[:_MAX_CANDIDATE_LEN]
    decoded: list[str] = []
    seen: set[str] = {text}

    def _add(candidate: str) -> bool:
        if len(decoded) >= _MAX_DECODE_CANDIDATES:
            return False
        candidate = candidate[:_MAX_CANDIDATE_LEN]
        if candidate in seen:
            return False
        seen.add(candidate)
        decoded.append(candidate)
        return True

    frontier = [text]
    for _depth in range(_MAX_DECODE_DEPTH):
        if len(decoded) >= _MAX_DECODE_CANDIDATES:
            break
        next_frontier: list[str] = []
        for layer_text in frontier:
            for candidate in _decode_one_layer(layer_text):
                if _add(candidate):
                    next_frontier.append(candidate)
                if len(decoded) >= _MAX_DECODE_CANDIDATES:
                    break
            if len(decoded) >= _MAX_DECODE_CANDIDATES:
                break
        if not next_frontier:
            break
        frontier = next_frontier

    for candidate in _decode_reversed(text):
        _add(candidate)

    # ROT13 — only flag if it surfaces a known jailbreak phrase.
    rot = codecs.encode(text, "rot_13")
    if _CIPHER_JAILBREAK_RE.search(rot):
        _add(rot)

    return decoded


# ---------------------------------------------------------------------------
# IngestionResult
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Structured output of the ingestion stage (Step 1).

    Attributes
    ----------
    original:
        The raw, unmodified prompt as received.
    normalized:
        Fully normalised text (invisible chars removed, homoglyphs replaced,
        whitespace injections collapsed, leetspeak translated).  This is what
        all downstream stages operate on.
    signature_hits:
        Jailbreak regex patterns that matched the normalised text or decoded
        payloads.
    decoded_payloads:
        Strings extracted from base64 / hex / ROT13 encodings found in the
        prompt.
    nearest_attack:
        The closest known-attack string in the corpus by similarity score.
    similarity:
        Cosine similarity score (semantic) or difflib ratio (fallback).
    homoglyph_detected:
        True if any homoglyph substitution was performed during normalisation.
    leetspeak_detected:
        True if any leetspeak substitution was performed.
    whitespace_injection_detected:
        True if whitespace-injected single-char runs were collapsed.
    variants:
        The text after each normalisation pass, in order: raw (NFKC +
        invisible-stripped), homoglyph-fixed, whitespace-defused,
        leet-normalized. ``normalized`` is the whitespace-defused variant
        (not leet-normalized — see below); signature matching in
        :func:`ingest` and the rule scans in :mod:`guard.detector` check
        every variant so a leetspeak-obfuscated attack is still caught via
        the leet-normalized variant without corrupting ordinary numeric
        text (``Q1``, ``60kg``) in the text everything else operates on.
    unsupported_language:
        True if the prompt shows no reliable sign of being English (see
        :func:`is_unsupported_language`). This product only supports
        English; such prompts are a decisive signal (folded into
        ``flagged``) since none of the English-pattern rules below can be
        trusted to have scanned them correctly.
    """

    original: str
    normalized: str
    signature_hits: list[str] = field(default_factory=list)
    decoded_payloads: list[str] = field(default_factory=list)
    nearest_attack: str = ""
    similarity: float = 0.0
    homoglyph_detected: bool = False
    leetspeak_detected: bool = False
    whitespace_injection_detected: bool = False
    unsupported_language: bool = False
    variants: list[str] = field(default_factory=list)

    # ── Canonical text surfaces (computed on-demand) ──────────────────────

    @property
    def compact(self) -> str:
        """Normalized text lowercased with all whitespace compressed to a single space."""
        return re.sub(r'\s+', ' ', self.normalized.lower()).strip()

    @property
    def alphanumeric(self) -> str:
        """Compact text with all non-[a-z0-9 ] characters removed.

        Useful for structural phrase matching: separators, punctuation, and
        special characters are collapsed to spaces before comparison.
        """
        stripped = re.sub(r'[^a-z0-9 ]', ' ', self.compact)
        return re.sub(r'\s+', ' ', stripped).strip()

    @property
    def decoded_text(self) -> str:
        """Joined text of all base64 / hex / ROT13 decoded payload fragments."""
        return ' '.join(self.decoded_payloads).lower()

    @property
    def url_decoded(self) -> str:
        """Compact text after URL/percent-decode — catches %XX obfuscation."""
        try:
            return urllib.parse.unquote(self.compact)
        except Exception:
            return self.compact

    @property
    def flagged(self) -> bool:
        """True when any decisive pre-detection attack signal is present."""
        return (
            bool(self.signature_hits or self.decoded_payloads)
            or self.similarity >= _SIMILARITY_THRESHOLD
            or self.unsupported_language
        )


# ---------------------------------------------------------------------------
# Introspection helper (used by the frontend to visualise the cascade)
# ---------------------------------------------------------------------------


def normalization_stages(prompt: str) -> list[tuple[str, str]]:
    """Return the text after each normalisation pass, in order.

    Each tuple is ``(pass_name, text_after_pass)``.  Purely for explainability
    — it lets a UI show exactly how a disguised prompt is peeled back to its
    canonical form.  The detector operates on :func:`ingest`'s ``normalized``
    field, which equals the last stage returned here (before ``.strip()``).
    """
    after_invisible = _strip_invisibles(prompt)
    after_homoglyph = _apply_homoglyph(after_invisible)
    after_ws = _defuse_whitespace_injection(after_homoglyph)
    after_leet = _apply_leetspeak(after_ws)
    return [
        ("raw input", prompt),
        ("strip invisibles + NFKC", after_invisible),
        ("homoglyph substitution", after_homoglyph),
        ("whitespace de-injection", after_ws),
        ("leetspeak normalisation", after_leet),
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def ingest(prompt: str) -> IngestionResult:
    """Normalise, decode, and pre-screen *prompt* before the detector runs.

    Applies normalisation passes in order, decodes encoded payloads, matches
    jailbreak signatures, and computes semantic (or difflib) similarity against
    the known-attack corpus.  All of this is deterministic and runs without any
    LLM call.
    """
    # ── Normalisation cascade ────────────────────────────────────────────
    after_invisible = _strip_invisibles(prompt).strip()

    after_homoglyph = _apply_homoglyph(after_invisible)
    homoglyph_detected = after_homoglyph != after_invisible

    after_ws = _defuse_whitespace_injection(after_homoglyph)
    whitespace_injection_detected = after_ws != after_homoglyph

    # ── Encoded payload detection (before leetspeak to preserve valid tokens) ──
    # base64 / hex tokens contain digits (0, 3, 5 …) that leetspeak would corrupt,
    # so we decode first, then apply the leet pass to the plain text only.
    decoded = _try_decode(after_ws)

    after_leet = _apply_leetspeak(after_ws)
    leetspeak_detected = after_leet != after_ws

    # `normalized` is the whitespace-defused variant, NOT leet-normalized:
    # leet substitution corrupts ordinary numeric text ("Q1" -> "Qi", "60kg"
    # -> "gokg"), so it must never be the primary text that detector.py's
    # rule scans and SAFE_MARKERS/STRONG_SAFE_FRAMES checks operate on
    # (P2-3). The leet-normalized text is still scanned for signatures
    # below (and exposed via `variants`) so a leetspeak-obfuscated attack
    # is still caught.
    normalized = after_ws.strip()
    variants = [after_invisible, after_homoglyph, after_ws, after_leet]

    # ── Signature matching (every normalisation variant + decoded payloads) ──
    haystacks = variants + decoded
    hits: list[str] = []
    for hay in haystacks:
        for pattern in _JAILBREAK_SIGNATURES:
            if re.search(pattern, hay, re.IGNORECASE) and pattern not in hits:
                hits.append(pattern)
        for pattern in _JAILBREAK_SIGNATURES_CASE_SENSITIVE:
            if re.search(pattern, hay) and pattern not in hits:
                hits.append(pattern)

    # ── Similarity (semantic embeddings with difflib fallback) ───────────
    # Lazy import to avoid circular dependency (semantic imports nothing from guard).
    from .semantic import ATTACK_CORPUS, difflib_similarity, get_matcher

    # Compute similarity on `normalized` (pre-leetspeak). Leetspeak (e.g. 60kg -> gokg)
    # destroys pretrained word embeddings, falsely lowering the similarity score.
    matcher = get_matcher()
    if matcher is not None:
        _, sim_score, nearest = matcher.match(normalized)
    else:
        sim_score, nearest = difflib_similarity(normalized, ATTACK_CORPUS)

    # ── Language check (on the pre-leetspeak text, same reasoning as similarity) ──
    unsupported_language = is_unsupported_language(normalized)

    return IngestionResult(
        original=prompt,
        normalized=normalized,
        signature_hits=hits,
        decoded_payloads=decoded,
        nearest_attack=nearest,
        similarity=sim_score,
        homoglyph_detected=homoglyph_detected,
        leetspeak_detected=leetspeak_detected,
        whitespace_injection_detected=whitespace_injection_detected,
        unsupported_language=unsupported_language,
        variants=variants,
    )
