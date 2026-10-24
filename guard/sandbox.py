"""Step 5 — Safe Execution Sandbox.

Executes the validated safe prompt and filters the output for PII/PHI or harmful
content before it reaches the user (deck: "apply an output filter for PII leakage
or harmful content"). Actual generation is delegated to Ollama when available; a
placeholder response is returned otherwise so the pipeline is demonstrable
offline. The output filter runs regardless of how the response was produced.
"""

import re
from typing import Callable, Optional

from .llm import OllamaClient
from .schemas import SandboxResult

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_MRN_RE = re.compile(r"\bMRN[:#]?\s*\d{4,}\b")
_DOB_RE = re.compile(r"\b(?:DOB|date of birth)[:\s]+\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b", re.IGNORECASE)

# Phone candidate: optional leading '+', 10-15 digits with optional
# separators. Deliberately loose — `_looks_like_phone` below rejects a bare
# contiguous digit run with neither a '+' nor a separator (an order ID, an
# account number, ...), which the old pattern used to flag unconditionally.
_PHONE_CANDIDATE_RE = re.compile(r"(?<!\d)\+?(?:\d[\s.-]?){9,15}\d(?!\d)")

# Credit-card candidate: 13-19 digits with optional grouping spaces/dashes.
# `_looks_like_card` below requires the digits to also pass the Luhn
# checksum, so an arbitrary numeric ID of the right length is not flagged.
_CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

# IBAN: 2-letter country code + 2 check digits + an 11-30 character
# alphanumeric BBAN (real IBANs run 15-34 chars total) — format-only match,
# no mod-97 checksum (the ticket only calls for a Luhn check, on cards).
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")

_EXEC_SYSTEM = """You are a compliant pharma assistant. Answer the user's request
using only aggregate/de-identified information and approved, non-promotional
language. Never include patient identifiers or medical/dosing advice."""


def _only_digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _looks_like_phone(match_text: str) -> bool:
    """10-15 digits, and either a leading '+' or at least one separator —
    a bare digit run with neither (an order ID, a numeric ID, ...) is not
    a plausible phone number (P2-5)."""
    digits = _only_digits(match_text)
    if not (10 <= len(digits) <= 15):
        return False
    stripped = match_text.strip()
    return stripped.startswith("+") or any(c in stripped for c in " .-")


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _looks_like_card(match_text: str) -> bool:
    digits = _only_digits(match_text)
    return 13 <= len(digits) <= 19 and _luhn_valid(digits)


# (label, compiled regex, optional validator). When a validator is given,
# a regex match is only redacted if validator(matched_text) is also True —
# needed for phone/card, whose regexes alone are too permissive (P2-5).
_PII_PATTERNS: list[tuple[str, "re.Pattern[str]", Optional[Callable[[str], bool]]]] = [
    ("email", _EMAIL_RE, None),
    ("phone", _PHONE_CANDIDATE_RE, _looks_like_phone),
    ("ssn", _SSN_RE, None),
    ("mrn", _MRN_RE, None),
    ("dob", _DOB_RE, None),
    ("card", _CARD_CANDIDATE_RE, _looks_like_card),
    ("iban", _IBAN_RE, None),
]


def _redact(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    redacted = text
    for label, pattern, validator in _PII_PATTERNS:
        def _sub(m: "re.Match[str]", _label: str = label, _validator=validator) -> str:
            if _validator is not None and not _validator(m.group(0)):
                return m.group(0)
            if _label not in found:
                found.append(_label)
            return f"[REDACTED_{_label.upper()}]"

        redacted = pattern.sub(_sub, redacted)
    return redacted, found


class SafeSandbox:
    def __init__(self, llm: Optional[OllamaClient] = None):
        self.llm = llm

    def execute(self, safe_prompt: str) -> SandboxResult:
        if self.llm is not None:
            raw = self.llm.generate_json(
                _EXEC_SYSTEM,
                safe_prompt + '\nRespond ONLY as JSON: {"response": "<answer>"}',
            )
            response = (raw or {}).get("response") if raw else None
            if not response:
                response = self._placeholder(safe_prompt)
        else:
            response = self._placeholder(safe_prompt)

        redacted, found = _redact(response if isinstance(response, str) else str(response))
        return SandboxResult(response=redacted, output_filtered=bool(found), filtered_items=found)

    @staticmethod
    def _placeholder(safe_prompt: str) -> str:
        return ("[sandbox stub — no LLM configured] Would execute the validated safe "
                f"prompt: {safe_prompt}")
