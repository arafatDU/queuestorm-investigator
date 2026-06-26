"""Deterministic complaint investigator.

This module implements :class:`TransactionInvestigator`, the pure-Python logic
layer that scores a complaint against a snippet of transaction history and
decides ``case_type``, ``evidence_verdict``, ``severity``, ``department``,
``human_review_required`` and supporting ``reason_codes``.

Design constraints (per ``AGENT.md`` §4 and §5.1):

* No LLM calls, no I/O, no globals — easy to unit-test.
* Handles English, Bangla, and mixed Banglish complaints.
* Bangla digits are normalized to ASCII before any numeric matching.
* Returns a typed :class:`InvestigationResult` so the LLM layer / API layer
  can simply forward fields onto the response model.

The class exposes a single entry-point — :meth:`TransactionInvestigator.investigate`
— that accepts the raw request payload (a ``dict`` or a ``AnalyzeTicketRequest``)
and returns a fully populated :class:`InvestigationResult`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from app.models.schemas import (
    AnalyzeTicketRequest,
    CaseType,
    Department,
    EvidenceVerdict,
    Severity,
    TransactionRecord,
)


# ---------------------------------------------------------------------------
# Constants and lookup tables
# ---------------------------------------------------------------------------


# High-value threshold (BDT) above which severity is bumped to high.
HIGH_VALUE_THRESHOLD_BDT: float = 10_000.0

# Amount-equality tolerance (BDT) for matching complaint numbers to history.
AMOUNT_TOLERANCE_BDT: float = 1.0

# Time window (seconds) used to detect duplicate transactions.
DUPLICATE_TIME_WINDOW_SECONDS: int = 60

# Bangla → ASCII digit map (U+09E6 .. U+09EF).
_BANGLA_DIGIT_MAP = {
    "০": "0",
    "১": "1",
    "২": "2",
    "৩": "3",
    "৪": "4",
    "৫": "5",
    "৬": "6",
    "৭": "7",
    "৮": "8",
    "৯": "9",
}


def _normalize_digits(text: str) -> str:
    """Convert Bangla digits to ASCII and normalize the rest of the string."""

    if not text:
        return ""
    out = []
    for ch in text:
        out.append(_BANGLA_DIGIT_MAP.get(ch, ch))
    normalized = "".join(out)
    # Strip zero-width / combining marks that show up in copy-pasted Bangla.
    return unicodedata.normalize("NFKC", normalized)


# Keyword tables — checked case-insensitively against the normalized complaint.
# Each pattern is a list of substrings; any match fires the rule.

PHISHING_KEYWORDS_EN = (
    "otp",
    "pin",
    "password",
    "passcode",
    "cvv",
    "card number",
    "share your",
    "send your",
    "verify your account",
    "call claiming to be",
    "someone called",
    "someone messaged",
    "asking for my otp",
    "asking for my pin",
    "asked for my otp",
    "asked for my pin",
    "social engineering",
)

PHISHING_KEYWORDS_BN = (
    "ওটিপি",
    "পিন",
    "পাসওয়ার্ড",
    "চাওয়া হয়েছে",
    "শেয়ার করতে",
    "জানতে চেয়েছে",
    "বলেছে অ্যাকাউন্ট বন্ধ",
)

WRONG_TRANSFER_KEYWORDS_EN = (
    "wrong number",
    "wrong person",
    "wrong recipient",
    "wrong account",
    "sent to the wrong",
    "mistakenly sent",
    "by mistake",
    "sent by mistake",
    "didn't receive",
    "did not receive",
    "not received",
    "hasn't received",
    "has not received",
    "he didn't get",
    "she didn't get",
    "brother didn't get",
    "sister didn't get",
    "friend didn't get",
    "person isn't responding",
    "isn't responding",
)

WRONG_TRANSFER_KEYWORDS_BN = (
    "ভুল নম্বর",
    "ভুল লোক",
    "ভুল রিসিভার",
    "ভুল ব্যক্তি",
    "ভুল একাউন্ট",
    "পাইনি",
    "পায়নি",
    "পাননি",
)

PAYMENT_FAILED_KEYWORDS_EN = (
    "payment failed",
    "failed payment",
    "payment didn't go through",
    "payment did not go through",
    "didn't go through",
    "did not go through",
    "wasn't successful",
    "was not successful",
    "not successful",
    "transaction failed",
    "balance was deducted",
    "balance is deducted",
    "balance deducted",
    "money was deducted",
    "money is deducted",
    "money deducted",
    "amount deducted",
    "charged but",
    "deducted twice",
    "but failed",
    "showed failed",
    "the app showed failed",
    "still deducted",
)

PAYMENT_FAILED_KEYWORDS_BN = (
    "ব্যালেন্স কেটে",
    "টাকা কেটে",
    "কেটে নিয়েছে",
    "কাটা হয়েছে",
    "পেমেন্ট ব্যর্থ",
    "লেনদেন ব্যর্থ",
    "ব্যর্থ হয়েছে",
)

DUPLICATE_KEYWORDS_EN = (
    "twice",
    "two times",
    "two times",
    "deducted twice",
    "charged twice",
    "duplicate payment",
    "duplicate charge",
    "duplicate transaction",
    "double charged",
    "paid twice",
    "charged two times",
)

DUPLICATE_KEYWORDS_BN = (
    "দুইবার",
    "দুই বার",
    "ডুপ্লিকেট",
)

MERCHANT_SETTLEMENT_KEYWORDS_EN = (
    "settlement",
    "settle",
    "settled",
    "merchant settlement",
    "not been settled",
    "haven't been settled",
    "haven't received my settlement",
    "merchant payout",
    "my sales",
    "yesterday's sales",
)

MERCHANT_SETTLEMENT_KEYWORDS_BN = (
    "সেটেলমেন্ট",
    "সেটলমেন্ট",
    "মার্চেন্ট",
    "বিক্রি",
)

AGENT_CASH_IN_KEYWORDS_EN = (
    "cash in",
    "cash-in",
    "cashin",
    "deposit through an agent",
    "deposited via agent",
    "agent cash in",
    "agent didn't credit",
    "agent didn't add",
    "agent didn't send",
    "agent said",
    "agent claims",
)

AGENT_CASH_IN_KEYWORDS_BN = (
    "ক্যাশ ইন",
    "ক্যাশইন",
    "এজেন্টের কাছে",
    "এজেন্ট বলছে",
)

REFUND_KEYWORDS_EN = (
    "refund",
    "refunded",
    "want my money back",
    "get my money back",
    "return my money",
    "money back",
    "reimburse",
    "reverse the payment",
    "reverse it",
    "please refund",
    "i want a refund",
)

REFUND_KEYWORDS_BN = (
    "ফেরত",
    "টাকা ফেরত",
    "ফেরত দিন",
    "রিফান্ড",
)

# Signals that should NOT be misclassified as duplicates or wrong transfers.
VAGUE_KEYWORDS_EN = (
    "something is wrong",
    "something wrong",
    "problem with my money",
    "issue with my account",
    "my money",
    "help me",
    "please check",
    "please help",
)


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------


# Amount with optional thousands separator + optional currency hint.
_AMOUNT_PATTERN = re.compile(
    r"""
    (?P<amount>\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?  # 1,234 or 1 234 or 1,234.56
    |\d+(?:\.\d+)?                                # 1234 or 1234.56
    )
    \s*
    (?P<currency>taka|tk|bdt|bdt|টাকা|৳)?
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Bangladesh phone numbers in any common local/international format.
_PHONE_PATTERN = re.compile(
    r"""
    (?<!\d)
    (?:\+?88)?
    0?
    1[3-9]
    \d{8}
    (?!\d)
    """,
    re.VERBOSE,
)

# Transaction IDs such as TXN-9101, txn_9101, TXN10001.
_TXN_PATTERN = re.compile(r"\b[Tt][Xx][Nn][-_\s]?(\d{2,8})\b")

# Currency/keyword cue — used to disambiguate numbers from non-amount numerics.
_CURRENCY_CUE = re.compile(
    r"\b(taka|tk|bdt|invoice|amount|paid|sent|received|deposit|cash)\b|টাকা",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExtractedSignals:
    """Raw signals pulled from the complaint text."""

    amounts: list[float] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    txn_ids: list[str] = field(default_factory=list)
    language_hint: str = "en"  # "en", "bn", or "mixed"
    has_phishing_signal: bool = False
    has_wrong_transfer_signal: bool = False
    has_payment_failed_signal: bool = False
    has_duplicate_signal: bool = False
    has_merchant_settlement_signal: bool = False
    has_agent_cash_in_signal: bool = False
    has_refund_signal: bool = False
    is_vague: bool = False


@dataclass(slots=True)
class MatchedTransaction:
    """Lightweight handle to a history entry that scored as a candidate."""

    transaction_id: str
    match_reason: str  # "txn_id" | "amount" | "phone"
    score: int         # higher = stronger signal
    record: TransactionRecord


@dataclass(slots=True)
class InvestigationResult:
    """The deterministic output of :class:`TransactionInvestigator`."""

    relevant_transaction_id: Optional[str]
    evidence_verdict: EvidenceVerdict
    case_type: CaseType
    severity: Severity
    department: Department
    human_review_required: bool
    reason_codes: list[str]
    signals: ExtractedSignals
    matched_transaction: Optional[TransactionRecord] = None

    # Convenience for tests / logs
    def as_dict(self) -> dict[str, Any]:
        return {
            "relevant_transaction_id": self.relevant_transaction_id,
            "evidence_verdict": self.evidence_verdict.value,
            "case_type": self.case_type.value,
            "severity": self.severity.value,
            "department": self.department.value,
            "human_review_required": self.human_review_required,
            "reason_codes": list(self.reason_codes),
        }


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------


def _has_any(haystack: str, needles: Iterable[str]) -> bool:
    """Case-insensitive substring match; needles may include Bangla substrings."""

    lowered = haystack.lower()
    for needle in needles:
        if not needle:
            continue
        if needle.lower() in lowered:
            return True
    return False


def _is_vague(complaint_lower: str, signals: ExtractedSignals) -> bool:
    """A complaint is vague when it carries no amount, phone, or TXN id."""

    if (
        not signals.amounts
        and not signals.phones
        and not signals.txn_ids
    ):
        # Vague only when no actionable signal AND no specific keyword pattern.
        if _has_any(complaint_lower, VAGUE_KEYWORDS_EN):
            return True
        # If the complaint is < 5 words it's almost certainly vague.
        return len(complaint_lower.split()) <= 5
    return False


def _normalize_phone(phone: str) -> str:
    """Return the last 10 digits of a BD phone number, normalized."""

    digits = re.sub(r"\D", "", _normalize_digits(phone))
    # Strip country code 880 and leading 0; keep last 10 digits.
    if digits.startswith("880") and len(digits) >= 13:
        digits = digits[3:]
    if digits.startswith("0"):
        digits = digits[1:]
    return digits[-10:] if len(digits) >= 10 else digits


def _phone_match(a: str, b: str) -> bool:
    """Compare two phone strings on the normalized last-10-digit form."""

    na, nb = _normalize_phone(a), _normalize_phone(b)
    return bool(na) and bool(nb) and na == nb


def _amount_match(a: float, b: float, tolerance: float = AMOUNT_TOLERANCE_BDT) -> bool:
    """Float-tolerant amount comparison."""

    return abs(a - b) <= tolerance


def _parse_amount(raw: str) -> Optional[float]:
    """Parse a matched amount group into a float, ignoring commas/spaces."""

    cleaned = re.sub(r"[,\s]", "", raw)
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main investigator class
# ---------------------------------------------------------------------------


class TransactionInvestigator:
    """Deterministic complaint investigator.

    Usage::

        investigator = TransactionInvestigator()
        result = investigator.investigate(complaint, transaction_history)
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        *,
        high_value_threshold: float = HIGH_VALUE_THRESHOLD_BDT,
        amount_tolerance: float = AMOUNT_TOLERANCE_BDT,
        duplicate_window_seconds: int = DUPLICATE_TIME_WINDOW_SECONDS,
    ) -> None:
        self.high_value_threshold = high_value_threshold
        self.amount_tolerance = amount_tolerance
        self.duplicate_window_seconds = duplicate_window_seconds

    # --------------------------------------------------------------- public

    def investigate(
        self,
        complaint: str,
        transaction_history: list[dict[str, Any] | TransactionRecord],
        *,
        user_type: Optional[str] = None,
        language_hint: Optional[str] = None,
    ) -> InvestigationResult:
        """Score the complaint and return a deterministic investigation result."""

        signals = self._extract_signals(complaint, language_hint=language_hint)
        candidates = self._match_candidates(signals, transaction_history)

        # If no numeric/phone signal matched but the complaint references a
        # category that aligns with the only record in history, adopt it as
        # the relevant transaction.
        if not candidates:
            fallback = self._fallback_single_history_match(
                signals, transaction_history, user_type=user_type
            )
            if fallback is not None:
                candidates = [fallback]

        records = [
            t if isinstance(t, TransactionRecord) else TransactionRecord.model_validate(t)
            for t in transaction_history
        ]
        matched, verdict, match_reason = self._decide_verdict(
            candidates, signals, records
        )
        case_type = self._classify_case(signals, matched, user_type=user_type)
        severity = self._decide_severity(case_type, verdict, matched, signals)
        department = self._route_department(case_type, verdict)
        review_required, reason_codes = self._decide_escalation(
            case_type=case_type,
            verdict=verdict,
            matched=matched,
            severity=severity,
            signals=signals,
            match_reason=match_reason,
            candidates_count=len(candidates),
        )

        return InvestigationResult(
            relevant_transaction_id=matched.transaction_id if matched else None,
            evidence_verdict=verdict,
            case_type=case_type,
            severity=severity,
            department=department,
            human_review_required=review_required,
            reason_codes=reason_codes,
            signals=signals,
            matched_transaction=matched,
        )

    # Convenience overload for the typed request model.
    def investigate_request(
        self, request: AnalyzeTicketRequest
    ) -> InvestigationResult:
        history: list[dict[str, Any] | TransactionRecord] = []
        for txn in request.transaction_history:
            if isinstance(txn, TransactionRecord):
                history.append(txn)
            else:
                history.append(TransactionRecord.model_validate(txn))

        language_hint = (
            request.language.value if request.language is not None else None
        )
        user_type = (
            request.user_type.value if request.user_type is not None else None
        )

        return self.investigate(
            complaint=request.complaint,
            transaction_history=history,
            user_type=user_type,
            language_hint=language_hint,
        )

    # --------------------------------------------------------- signal extract

    def _extract_signals(
        self,
        complaint: str,
        *,
        language_hint: Optional[str] = None,
    ) -> ExtractedSignals:
        normalized = _normalize_digits(complaint or "")
        complaint_lower = normalized.lower()

        # Amounts: scan regex hits, only keep those near a currency cue OR
        # plain standalone numbers (heuristic — most complaints mention an
        # amount either as "5000" with "taka" or as "$5000"-style figures).
        amounts: list[float] = []
        for match in _AMOUNT_PATTERN.finditer(normalized):
            raw = match.group("amount")
            value = _parse_amount(raw)
            if value is None:
                continue
            # Skip tiny numbers that are almost certainly not amounts (e.g. "I
            # tried 3 times"). Require at least 2 digits OR a currency cue.
            if value < 10 and not match.group("currency"):
                continue
            # Require proximity to a currency keyword OR a 3+ digit number.
            start, end = match.span()
            window = normalized[max(0, start - 20): min(len(normalized), end + 20)]
            if value >= 100 or _CURRENCY_CUE.search(window) or match.group("currency"):
                amounts.append(value)

        # Phones
        phones: list[str] = []
        for match in _PHONE_PATTERN.finditer(normalized):
            phone = match.group(0)
            if _normalize_phone(phone):
                phones.append(phone)

        # TXN IDs
        txn_ids: list[str] = []
        for match in _TXN_PATTERN.finditer(normalized):
            txn_ids.append(f"TXN-{match.group(1)}")

        # Language hint detection
        has_bn = any("\u0980" <= ch <= "\u09FF" for ch in normalized)
        has_en = bool(re.search(r"[a-zA-Z]", normalized))
        if language_hint in {"en", "bn", "mixed"}:
            detected_language = language_hint
        elif has_bn and has_en:
            detected_language = "mixed"
        elif has_bn:
            detected_language = "bn"
        else:
            detected_language = "en"

        signals = ExtractedSignals(
            amounts=amounts,
            phones=phones,
            txn_ids=txn_ids,
            language_hint=detected_language,
            is_vague=_is_vague(complaint_lower, ExtractedSignals(
                amounts=amounts, phones=phones, txn_ids=txn_ids
            )),
        )

        # Keyword presence flags
        signals.has_phishing_signal = (
            _has_any(complaint_lower, PHISHING_KEYWORDS_EN)
            or _has_any(normalized, PHISHING_KEYWORDS_BN)
        )
        signals.has_wrong_transfer_signal = (
            _has_any(complaint_lower, WRONG_TRANSFER_KEYWORDS_EN)
            or _has_any(normalized, WRONG_TRANSFER_KEYWORDS_BN)
        )
        signals.has_payment_failed_signal = (
            _has_any(complaint_lower, PAYMENT_FAILED_KEYWORDS_EN)
            or _has_any(normalized, PAYMENT_FAILED_KEYWORDS_BN)
        )
        signals.has_duplicate_signal = (
            _has_any(complaint_lower, DUPLICATE_KEYWORDS_EN)
            or _has_any(normalized, DUPLICATE_KEYWORDS_BN)
        )
        signals.has_merchant_settlement_signal = (
            _has_any(complaint_lower, MERCHANT_SETTLEMENT_KEYWORDS_EN)
            or _has_any(normalized, MERCHANT_SETTLEMENT_KEYWORDS_BN)
        )
        signals.has_agent_cash_in_signal = (
            _has_any(complaint_lower, AGENT_CASH_IN_KEYWORDS_EN)
            or _has_any(normalized, AGENT_CASH_IN_KEYWORDS_BN)
        )
        signals.has_refund_signal = (
            _has_any(complaint_lower, REFUND_KEYWORDS_EN)
            or _has_any(normalized, REFUND_KEYWORDS_BN)
        )

        # Refine: phishing check requires credential request context, not just
        # the mere presence of the word "OTP" inside a description. We treat
        # the presence of phishing keywords as sufficient signal — it will be
        # overridden later if a clearer case_type keyword wins.
        return signals

    # ---------------------------------------------------------- matching

    def _match_candidates(
        self,
        signals: ExtractedSignals,
        history: list[dict[str, Any] | TransactionRecord],
    ) -> list[MatchedTransaction]:
        """Score each history entry against extracted signals."""

        records = [
            t if isinstance(t, TransactionRecord) else TransactionRecord.model_validate(t)
            for t in history
        ]

        candidates: list[MatchedTransaction] = []

        # --- TXN-ID match (strongest) ------------------------------------
        if signals.txn_ids:
            txn_lookup = {tid.lower(): rec for rec in records for tid in [rec.transaction_id.lower()]}
            for txn_id in signals.txn_ids:
                hit = txn_lookup.get(txn_id.lower())
                if hit is not None:
                    candidates.append(
                        MatchedTransaction(
                            transaction_id=hit.transaction_id,
                            match_reason="txn_id",
                            score=1000,
                            record=hit,
                        )
                    )
            if candidates:
                # TXN-ID match wins outright.
                return candidates

        # --- Amount + phone scoring -------------------------------------
        for rec in records:
            score = 0
            reasons: list[str] = []

            if rec.amount is not None:
                for amt in signals.amounts:
                    if _amount_match(rec.amount, amt, self.amount_tolerance):
                        score += 100
                        reasons.append("amount")
                        break

            if rec.counterparty:
                for phone in signals.phones:
                    if _phone_match(rec.counterparty, phone):
                        score += 200
                        reasons.append("phone")
                        break

            if score > 0:
                reason = "amount" if reasons == ["amount"] else (
                    "phone" if reasons == ["phone"] else "amount+phone"
                )
                candidates.append(
                    MatchedTransaction(
                        transaction_id=rec.transaction_id,
                        match_reason=reason,
                        score=score,
                        record=rec,
                    )
                )

        candidates.sort(key=lambda m: (-m.score, m.transaction_id))
        return candidates

    def _fallback_single_history_match(
        self,
        signals: ExtractedSignals,
        history: list[dict[str, Any] | TransactionRecord],
        *,
        user_type: Optional[str] = None,
    ) -> Optional[MatchedTransaction]:
        """When the history contains exactly one record AND the complaint
        references a category that fits that record (e.g. a single settlement
        for a merchant asking about settlement), treat it as a relevant
        candidate even when no numeric/phone signals match.

        This avoids returning ``insufficient_data`` for clear, unambiguous
        single-record complaints like "Settlement has not arrived for
        yesterday's sales" when only one settlement transaction exists.
        """

        records = [
            t if isinstance(t, TransactionRecord) else TransactionRecord.model_validate(t)
            for t in history
        ]
        if len(records) != 1:
            return None
        rec = records[0]
        is_merchant = (user_type or "").lower() == "merchant"

        # Merchant settlement / cash-in agent — single history record wins.
        if signals.has_merchant_settlement_signal and (
            is_merchant or (rec.type and rec.type.value == "settlement")
        ):
            return MatchedTransaction(
                transaction_id=rec.transaction_id,
                match_reason="single_record_fallback",
                score=10,
                record=rec,
            )
        if signals.has_agent_cash_in_signal and (
            rec.type is None or (rec.type and rec.type.value == "cash_in")
        ):
            return MatchedTransaction(
                transaction_id=rec.transaction_id,
                match_reason="single_record_fallback",
                score=10,
                record=rec,
            )
        return None

    # ---------------------------------------------------- verdict decision

    def _decide_verdict(
        self,
        candidates: list[MatchedTransaction],
        signals: ExtractedSignals,
        history: list[TransactionRecord],
    ) -> tuple[Optional[MatchedTransaction], EvidenceVerdict, str]:
        """Pick the matched transaction and the evidence verdict.

        Returns ``(matched, verdict, match_reason)``. ``match_reason`` is one
        of: ``no_match``, ``vague``, ``ambiguous_match``, ``duplicate_pair``,
        ``txn_id``, ``amount``, ``phone``, ``amount+phone``.
        """

        # No candidates → insufficient data (phishing is handled later).
        if not candidates:
            return None, EvidenceVerdict.INSUFFICIENT_DATA, "no_match"

        # Vague complaint → insufficient even when history matches weakly.
        if signals.is_vague and not signals.txn_ids and not signals.phones:
            return None, EvidenceVerdict.INSUFFICIENT_DATA, "vague"

        # Multiple candidates → ambiguous unless the duplicate-payment
        # detector can resolve them deterministically.
        if len(candidates) > 1:
            dup = self._detect_duplicate_pair(candidates)
            if dup is not None:
                return dup, EvidenceVerdict.CONSISTENT, "duplicate_pair"
            return None, EvidenceVerdict.INSUFFICIENT_DATA, "ambiguous_match"

        match = candidates[0]

        # Inconsistency: a "wrong number / wrong person" claim is contradicted
        # when the supplied history shows ≥ 1 *other* completed transfer to
        # the same counterparty (i.e. an established recipient relationship).
        if (
            signals.has_wrong_transfer_signal
            and match.record.counterparty
            and match.record.type is not None
            and match.record.type.value == "transfer"
            and match.record.status is not None
            and match.record.status.value == "completed"
        ):
            same_counterparty = sum(
                1
                for rec in history
                if rec.transaction_id != match.transaction_id
                and rec.counterparty
                and _phone_match(rec.counterparty, match.record.counterparty)
                and rec.type is not None
                and rec.type.value == "transfer"
                and rec.status is not None
                and rec.status.value == "completed"
            )
            if same_counterparty >= 1:
                return match, EvidenceVerdict.INCONSISTENT, match.match_reason

        return match, EvidenceVerdict.CONSISTENT, match.match_reason

    @staticmethod
    def _detect_duplicate_pair(
        candidates: list[MatchedTransaction],
    ) -> Optional[MatchedTransaction]:
        """If two candidates look like a duplicate pair, return the later one.

        A duplicate pair is two history entries with the same amount and same
        counterparty (case-insensitive) that occurred within
        ``duplicate_window_seconds`` of each other. The *later* transaction is
        the suspected duplicate and is returned as the relevant match.
        """

        if len(candidates) < 2:
            return None

        # Build (timestamp, candidate) pairs; skip entries without timestamps.
        timestamped: list[tuple[Any, MatchedTransaction]] = []
        for cand in candidates:
            ts = cand.record.timestamp
            if ts is not None:
                timestamped.append((ts, cand))
        if len(timestamped) < 2:
            return None

        timestamped.sort(key=lambda t: t[0])
        for i in range(len(timestamped) - 1):
            earlier_ts, earlier = timestamped[i]
            later_ts, later = timestamped[i + 1]
            try:
                delta = (later_ts - earlier_ts).total_seconds()
            except AttributeError:
                # datetime already has total_seconds; if it raises we can't
                # make a duplicate determination from this pair.
                continue
            if delta < 0:
                continue
            if delta > TransactionInvestigator._duplicate_window(candidates):
                continue
            if (
                earlier.record.amount is not None
                and later.record.amount is not None
                and _amount_match(earlier.record.amount, later.record.amount)
                and earlier.record.counterparty
                and later.record.counterparty
                and earlier.record.counterparty.strip().lower()
                == later.record.counterparty.strip().lower()
            ):
                return later
        return None

    @staticmethod
    def _duplicate_window(candidates: list[MatchedTransaction]) -> int:
        """Resolve the duplicate time window from the first candidate if set."""

        # The candidates were constructed with the class-level window; pull it
        # back so the helper stays self-contained.
        return DUPLICATE_TIME_WINDOW_SECONDS

    # ------------------------------------------------- case_type / severity

    def _classify_case(
        self,
        signals: ExtractedSignals,
        matched: Optional[MatchedTransaction],
        *,
        user_type: Optional[str] = None,
    ) -> CaseType:
        """Pick a case_type using priority-ordered keyword + signal rules."""

        # 1. Phishing always wins (safety-critical).
        if signals.has_phishing_signal:
            return CaseType.PHISHING_OR_SOCIAL_ENGINEERING

        # 2. Merchant settlement (only when matched transaction is settlement
        #    OR the complaint is clearly merchant-side and mentions settlement).
        is_merchant = (user_type or "").lower() == "merchant"
        if (
            signals.has_merchant_settlement_signal
            and (is_merchant or (matched and matched.record.type and matched.record.type.value == "settlement"))
        ):
            return CaseType.MERCHANT_SETTLEMENT_DELAY

        # 3. Agent cash-in.
        if signals.has_agent_cash_in_signal and (
            matched is None or (matched.record.type and matched.record.type.value == "cash_in")
        ):
            return CaseType.AGENT_CASH_IN_ISSUE

        # 4. Duplicate payment — requires either an explicit duplicate keyword
        #    OR detection of two near-simultaneous same-amount same-merchant
        #    entries in the supplied history.
        if signals.has_duplicate_signal:
            return CaseType.DUPLICATE_PAYMENT

        # 5. Wrong transfer.
        if signals.has_wrong_transfer_signal:
            return CaseType.WRONG_TRANSFER

        # 6. Payment failed.
        if signals.has_payment_failed_signal and (
            matched is None
            or (matched.record.type and matched.record.type.value in {"payment", "transfer", "cash_in"})
            or (matched.record.status and matched.record.status.value == "failed")
        ):
            return CaseType.PAYMENT_FAILED

        # 7. Refund request (only when there is a real candidate — otherwise
        #    treat the refund mention as noise).
        if signals.has_refund_signal and matched is not None:
            return CaseType.REFUND_REQUEST

        # 8. Vague → other.
        return CaseType.OTHER

    def _decide_severity(
        self,
        case_type: CaseType,
        verdict: EvidenceVerdict,
        matched: Optional[MatchedTransaction],
        signals: ExtractedSignals,
    ) -> Severity:
        """Map case_type + verdict + amount to a severity tier."""

        # Critical — phishing is always critical.
        if case_type == CaseType.PHISHING_OR_SOCIAL_ENGINEERING:
            return Severity.CRITICAL

        amount = matched.record.amount if matched and matched.record.amount is not None else None
        is_high_value = amount is not None and amount >= self.high_value_threshold

        # High — clear wrong_transfer with consistent evidence, duplicate, or
        # agent cash-in issue (with pending status).
        if case_type == CaseType.WRONG_TRANSFER and verdict == EvidenceVerdict.CONSISTENT:
            return Severity.HIGH

        if case_type == CaseType.DUPLICATE_PAYMENT:
            return Severity.HIGH

        if (
            case_type == CaseType.AGENT_CASH_IN_ISSUE
            and matched is not None
            and matched.record.status is not None
            and matched.record.status.value == "pending"
        ):
            return Severity.HIGH

        if case_type == CaseType.PAYMENT_FAILED and (
            is_high_value
            or (matched and matched.record.status and matched.record.status.value == "failed")
            or (matched and matched.record.status and matched.record.status.value == "pending")
        ):
            return Severity.HIGH

        if is_high_value:
            return Severity.HIGH

        # Medium — merchant settlements, inconsistent, ambiguous.
        if case_type == CaseType.MERCHANT_SETTLEMENT_DELAY:
            return Severity.MEDIUM
        if verdict == EvidenceVerdict.INCONSISTENT:
            return Severity.MEDIUM
        if verdict == EvidenceVerdict.INSUFFICIENT_DATA:
            # Vague complaints are LOW; everything else insufficient → medium.
            if signals.is_vague:
                return Severity.LOW
            return Severity.MEDIUM

        # Low — refund requests with consistent evidence, vague/other.
        if case_type == CaseType.REFUND_REQUEST:
            return Severity.LOW
        if case_type == CaseType.OTHER:
            return Severity.LOW

        return Severity.MEDIUM

    # ------------------------------------------------------ routing

    def _route_department(
        self, case_type: CaseType, verdict: EvidenceVerdict
    ) -> Department:
        """Map case_type (+ verdict) to the team that should own the ticket."""

        if case_type == CaseType.PHISHING_OR_SOCIAL_ENGINEERING:
            return Department.FRAUD_RISK
        if case_type == CaseType.WRONG_TRANSFER:
            return Department.DISPUTE_RESOLUTION
        if case_type in {CaseType.PAYMENT_FAILED, CaseType.DUPLICATE_PAYMENT}:
            return Department.PAYMENTS_OPS
        if case_type == CaseType.MERCHANT_SETTLEMENT_DELAY:
            return Department.MERCHANT_OPERATIONS
        if case_type == CaseType.AGENT_CASH_IN_ISSUE:
            return Department.AGENT_OPERATIONS
        if case_type == CaseType.REFUND_REQUEST:
            # Contested refunds go to dispute_resolution; otherwise general support.
            return (
                Department.DISPUTE_RESOLUTION
                if verdict == EvidenceVerdict.INCONSISTENT
                else Department.CUSTOMER_SUPPORT
            )
        # other / vague
        return Department.CUSTOMER_SUPPORT

    # --------------------------------------------------- escalation logic

    def _decide_escalation(
        self,
        *,
        case_type: CaseType,
        verdict: EvidenceVerdict,
        matched: Optional[MatchedTransaction],
        severity: Severity,
        signals: ExtractedSignals,
        match_reason: str,
        candidates_count: int,
    ) -> tuple[bool, list[str]]:
        """Compute human_review_required and the supporting reason codes."""

        reasons: list[str] = []
        review = False

        if candidates_count == 0:
            if not signals.txn_ids and not signals.phones and not signals.amounts:
                reasons.append("no_match")
            else:
                reasons.append("no_match")
        elif candidates_count > 1:
            reasons.append("ambiguous_match")
        elif match_reason == "duplicate_pair":
            reasons.append("transaction_match_by_amount")
            reasons.append("duplicate_pair_resolution")
        elif match_reason == "txn_id":
            reasons.append("transaction_match_by_txn_id")
        elif match_reason == "phone":
            reasons.append("transaction_match_by_phone")
        elif match_reason == "amount":
            reasons.append("transaction_match_by_amount")
        elif match_reason == "amount+phone":
            reasons.extend(["transaction_match_by_amount", "transaction_match_by_phone"])

        # --- Inconsistency detection (needs the full history) -------------
        if (
            verdict == EvidenceVerdict.CONSISTENT
            and case_type == CaseType.WRONG_TRANSFER
            and matched is not None
        ):
            # Caller will refine verdict via investigate(); we don't have the
            # full history here so leave it as consistent and let the wrapper
            # patch this up.
            pass

        # --- Keyword reason codes ----------------------------------------
        if case_type == CaseType.PHISHING_OR_SOCIAL_ENGINEERING:
            reasons.append("phishing_keywords")
        elif case_type == CaseType.WRONG_TRANSFER:
            reasons.append("wrong_transfer_keywords")
        elif case_type == CaseType.PAYMENT_FAILED:
            reasons.append("payment_failed_keywords")
        elif case_type == CaseType.DUPLICATE_PAYMENT:
            reasons.append("duplicate_keywords")
        elif case_type == CaseType.MERCHANT_SETTLEMENT_DELAY:
            reasons.append("merchant_settlement_keywords")
        elif case_type == CaseType.AGENT_CASH_IN_ISSUE:
            reasons.append("agent_cash_in_keywords")
        elif case_type == CaseType.REFUND_REQUEST:
            reasons.append("refund_keywords")

        # --- human_review_required decision ------------------------------
        # Ambiguous matches (multiple plausible transactions) do NOT auto-
        # escalate — the clarifying reply is the correct path. Review may
        # still be triggered by other rules below (phishing, critical, etc.).
        ambiguous = match_reason == "ambiguous_match"

        if verdict == EvidenceVerdict.INCONSISTENT:
            review = True
            reasons.append("evidence_inconsistent")

        if case_type == CaseType.PHISHING_OR_SOCIAL_ENGINEERING:
            review = True

        if not ambiguous and case_type in {
            CaseType.WRONG_TRANSFER,
            CaseType.DUPLICATE_PAYMENT,
            CaseType.AGENT_CASH_IN_ISSUE,
        }:
            review = True

        # Merchant settlement: escalate only when the settlement value is
        # large enough that merchant_operations should treat it as a priority
        # case rather than the standard reconciliation flow.
        if (
            case_type == CaseType.MERCHANT_SETTLEMENT_DELAY
            and matched is not None
            and matched.record.amount is not None
            and matched.record.amount >= 2 * self.high_value_threshold
        ):
            review = True

        if (
            case_type == CaseType.PAYMENT_FAILED
            and matched is not None
            and matched.record.amount is not None
            and matched.record.amount >= self.high_value_threshold
        ):
            review = True

        if matched is not None and matched.record.status is not None:
            status = matched.record.status.value
            if status == "pending":
                # Pending is the normal expected state for a settlement; only
                # escalate when it's a non-merchant case OR a very high-value
                # merchant transaction.
                is_low_value_merchant_settlement = (
                    case_type == CaseType.MERCHANT_SETTLEMENT_DELAY
                    and matched.record.amount is not None
                    and matched.record.amount < 2 * self.high_value_threshold
                )
                if not is_low_value_merchant_settlement:
                    review = True
                reasons.append("pending_transaction")
            # Note: failed payments do NOT auto-escalate; payments_ops handles
            # them through the standard reconciliation flow (SAMPLE-03).

        if severity in {Severity.CRITICAL, Severity.HIGH} and case_type in {
            CaseType.WRONG_TRANSFER,
            CaseType.AGENT_CASH_IN_ISSUE,
            CaseType.DUPLICATE_PAYMENT,
            CaseType.PHISHING_OR_SOCIAL_ENGINEERING,
        }:
            reasons.append("critical_escalation")

        if case_type == CaseType.PHISHING_OR_SOCIAL_ENGINEERING:
            reasons.append("human_review_required")

        # De-duplicate while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for code in reasons:
            if code and code not in seen:
                seen.add(code)
                deduped.append(code)

        return review, deduped


__all__ = [
    "AGENT_CASH_IN_KEYWORDS_BN",
    "AGENT_CASH_IN_KEYWORDS_EN",
    "DUPLICATE_KEYWORDS_BN",
    "DUPLICATE_KEYWORDS_EN",
    "ExtractedSignals",
    "InvestigationResult",
    "MatchedTransaction",
    "MERCHANT_SETTLEMENT_KEYWORDS_BN",
    "MERCHANT_SETTLEMENT_KEYWORDS_EN",
    "PAYMENT_FAILED_KEYWORDS_BN",
    "PAYMENT_FAILED_KEYWORDS_EN",
    "PHISHING_KEYWORDS_BN",
    "PHISHING_KEYWORDS_EN",
    "REFUND_KEYWORDS_BN",
    "REFUND_KEYWORDS_EN",
    "TransactionInvestigator",
    "WRONG_TRANSFER_KEYWORDS_BN",
    "WRONG_TRANSFER_KEYWORDS_EN",
]