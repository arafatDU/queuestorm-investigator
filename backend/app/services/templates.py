"""Deterministic template generator used as the LLM fallback.

When the LLM call fails (network error, timeout, missing API key, safety
block, or invalid JSON response) the API still needs to return a useful,
SAFE response in <30 seconds. This module produces the three string fields
(``agent_summary``, ``recommended_next_action``, ``customer_reply``) from
the deterministic :class:`InvestigationResult` plus a small library of
hardcoded safe templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.services.investigator import InvestigationResult
from app.services.safety import safe_fallback, safe_next_action


_REPLY_SAFE_BOILERPLATE_EN = (
    "Please do not share your PIN or OTP with anyone. We will update you "
    "through official support channels."
)
_REPLY_SAFE_BOILERPLATE_BN = (
    "অনুগ্রহ করে কারো সাথে আপনার পিন বা ওটিপি শেয়ার করবেন না। আমরা "
    "অফিসিয়াল সাপোর্ট চ্যানেলের মাধ্যমে আপনাকে জানাব।"
)


@dataclass(slots=True)
class TemplateReply:
    agent_summary: str
    recommended_next_action: str
    customer_reply: str
    language: str = "en"


def render_templates(
    result: InvestigationResult,
    *,
    language: Optional[str] = None,
) -> TemplateReply:
    lang = _resolve_language(language or result.signals.language_hint)
    boilerplate = _REPLY_SAFE_BOILERPLATE_BN if lang == "bn" else _REPLY_SAFE_BOILERPLATE_EN

    return TemplateReply(
        agent_summary=_build_agent_summary(result, lang),
        recommended_next_action=_build_next_action(result, lang),
        customer_reply=_build_customer_reply(result, lang, boilerplate),
        language=lang,
    )


def render_safe_only(language: str = "en") -> TemplateReply:
    lang = _resolve_language(language)
    return TemplateReply(
        agent_summary=(
            "Customer complaint received. Awaiting manual review by the support team."
            if lang == "en"
            else "গ্রাহকের অভিযোগ গ্রহণ করা হয়েছে। সাপোর্ট টিমের ম্যানুয়াল রিভিউ প্রক্রিয়াধীন।"
        ),
        recommended_next_action=safe_next_action(lang),
        customer_reply=safe_fallback(lang),
        language=lang,
    )


def _build_agent_summary(result: InvestigationResult, lang: str) -> str:
    case = _case_label(result.case_type.value, lang)
    txn = result.relevant_transaction_id or _none_label(lang)
    verdict = _verdict_label(result.evidence_verdict.value, lang)
    severity = _severity_label(result.severity.value, lang)
    dept = _department_label(result.department.value, lang)

    if lang == "bn":
        return (
            f"গ্রাহকের অভিযোগটি {case} হিসেবে শ্রেণীবদ্ধ করা হয়েছে "
            f"(তীব্রতা: {severity}, প্রমাণ: {verdict}, সম্পর্কিত লেনদেন: {txn}). "
            f"রাউটিং দল: {dept}."
        )
    return (
        f"Customer complaint classified as {case} (severity: {severity}, "
        f"evidence: {verdict}, related transaction: {txn}). Routing: {dept}."
    )


def _build_next_action(result: InvestigationResult, lang: str) -> str:
    case = result.case_type.value
    verdict = result.evidence_verdict.value
    txn = result.relevant_transaction_id
    review = result.human_review_required

    if lang == "bn":
        prefix = "পরবর্তী পদক্ষেপ"
        review_phrase = "মানব পর্যালোচনা প্রয়োজন" if review else "স্বয়ংক্রিয় রাউটিং"
    else:
        prefix = "Next step"
        review_phrase = "Human review required" if review else "Auto-routing"

    txn_part = f" ({txn})" if txn else ""
    return (
        f"{prefix} ({review_phrase}): review {case} case{txn_part} with "
        f"verdict={verdict}; follow the standard {result.department.value} "
        f"workflow and update the customer through official support channels. "
        f"Do not request any credential or promise a refund outside "
        f"official processes."
    )


def _build_customer_reply(result: InvestigationResult, lang: str, boilerplate: str) -> str:
    case = result.case_type.value
    verdict = result.evidence_verdict.value
    txn = result.relevant_transaction_id

    if verdict == "insufficient_data":
        if lang == "bn":
            base = (
                "আমরা আপনার অনুরোধ পেয়েছি। দ্রুত সহায়তা করতে, অনুগ্রহ করে "
                "সংশ্লিষ্ট লেনদেন আইডি, প্রাপকের নম্বর (যদি প্রযোজ্য হয়), "
                "এবং ঘটনার প্রায় সময় আমাদের জানান।"
            )
        else:
            base = (
                "Thank you for reaching out. To help you faster, please share "
                "the transaction ID, the recipient's phone number (if applicable), "
                "and the approximate time of the incident."
            )
    elif case == "phishing_or_social_engineering":
        if lang == "bn":
            base = (
                "শেয়ার করার আগে যোগাযোগ করার জন্য আপনাকে ধন্যবাদ। আমরা কখনোই "
                "আপনার পিন, ওটিপি বা পাসওয়ার্ড কোনো অবস্থাতেই চাই না। অনুগ্রহ "
                "করে এই তথ্য কারো সাথে শেয়ার করবেন না, এমনকি আমাদের নাম ধরে "
                "হলেও না। আমাদের ফ্রড টিম এই ঘটনাটি অবগত হয়েছে।"
            )
        else:
            base = (
                "Thank you for reaching out before sharing any information. "
                "We never ask for your PIN, OTP, or password under any "
                "circumstances. Please do not share these with anyone, even "
                "if they claim to be from us. Our fraud team has been notified "
                "of this incident."
            )
    elif case == "refund_request":
        if lang == "bn":
            base = (
                "আপনার অনুরোধ গ্রহণ করা হয়েছে। সম্পূর্ণ মার্চেন্ট পেমেন্টের ফেরত "
                "মার্চেন্টের নীতির উপর নির্ভর করে। প্রয়োজনে আমাদের জানান, আমরা "
                "আপনাকে সহায়তা করব।"
            )
        else:
            base = (
                "Thank you for reaching out. Refunds for completed merchant "
                "payments depend on the merchant's own policy. If you need "
                "help, please reply and we will guide you."
            )
    elif case == "merchant_settlement_delay":
        if lang == "bn":
            base = (
                "আপনার সেটেলমেন্ট অনুরোধ গ্রহণ করা হয়েছে। আমাদের মার্চেন্ট "
                "অপারেশন্স টিম ব্যাচের অবস্থা যাচাই করে অফিসিয়াল চ্যানেলের "
                "মাধ্যমে আপডেট জানাবে।"
            )
        else:
            base = (
                "We have noted your concern about the settlement. Our merchant "
                "operations team will check the batch status and update you "
                "on the expected settlement time through official channels."
            )
    elif case == "agent_cash_in_issue":
        suffix = f" {txn}" if txn else ""
        if lang == "bn":
            base = (
                f"আপনার লেনদেন{suffix} এর বিষয়ে আমরা অবগত হয়েছি। আমাদের "
                f"এজেন্ট অপারেশন্স দল এটি দ্রুত যাচাই করবে এবং অফিসিয়াল "
                f"চ্যানেলে আপনাকে জানাবে।"
            )
        else:
            base = (
                f"We have noted your concern about transaction{suffix}. "
                f"Our agent operations team will verify this and update you "
                f"through official channels."
            )
    elif case == "duplicate_payment":
        suffix = f" ({txn})" if txn else ""
        if lang == "bn":
            base = (
                f"সম্ভাব্য ডুপ্লিকেট লেনদেন{suffix} এর বিষয়ে আমরা অবগত হয়েছি। "
                f"আমাদের পেমেন্টস টিম বিলারের সাথে যাচাই করবে এবং যোগ্য কোনো "
                f"পরিমাণ অফিসিয়াল চ্যানেলের মাধ্যমে ফেরত দেওয়া হবে।"
            )
        else:
            base = (
                f"We have noted the possible duplicate payment{suffix}. "
                f"Our payments team will verify with the biller and any eligible "
                f"amount will be returned through official channels."
            )
    elif case == "payment_failed":
        suffix = f" {txn}" if txn else ""
        if lang == "bn":
            base = (
                f"লেনদেন{suffix} এর বিষয়ে আমরা অবগত হয়েছি। আমাদের পেমেন্টস "
                f"টিম এটি পর্যালোচনা করবে এবং যোগ্য কোনো পরিমাণ অফিসিয়াল "
                f"চ্যানেলের মাধ্যমে ফেরত দেওয়া হবে।"
            )
        else:
            base = (
                f"We have noted that transaction{suffix} may have caused an "
                f"unexpected balance deduction. Our payments team will review "
                f"the case and any eligible amount will be returned through "
                f"official channels."
            )
    elif case == "wrong_transfer":
        suffix = f" {txn}" if txn else ""
        if lang == "bn":
            base = (
                f"লেনদেন{suffix} এর বিষয়ে আমরা অবগত হয়েছি। আমাদের ডিসপিউট "
                f"টিম এটি পর্যালোচনা করবে এবং অফিসিয়াল সাপোর্ট চ্যানেলের "
                f"মাধ্যমে আপনাকে জানাবে।"
            )
        else:
            base = (
                f"We have noted your concern about transaction{suffix}. "
                f"Our dispute team will review the case and contact you through "
                f"official support channels."
            )
    else:
        if lang == "bn":
            base = (
                "আমরা আপনার অভিযোগ গ্রহণ করেছি। আমাদের সাপোর্ট টিম শীঘ্রই "
                "আপনার সাথে যোগাযোগ করবে।"
            )
        else:
            base = (
                "We have received your complaint. Our support team will get back "
                "to you shortly with an update."
            )

    return f"{base} {boilerplate}"


def _resolve_language(language: Optional[str]) -> str:
    return "bn" if language == "bn" else "en"


def _none_label(lang: str) -> str:
    return "পাওয়া যায়নি" if lang == "bn" else "none"


def _case_label(case_type: str, lang: str) -> str:
    en = {
        "wrong_transfer": "wrong transfer",
        "payment_failed": "failed payment",
        "refund_request": "refund request",
        "duplicate_payment": "duplicate payment",
        "merchant_settlement_delay": "merchant settlement delay",
        "agent_cash_in_issue": "agent cash-in issue",
        "phishing_or_social_engineering": "phishing / social engineering",
        "other": "other (unclassified)",
    }
    bn = {
        "wrong_transfer": "ভুল ট্রান্সফার",
        "payment_failed": "ব্যর্থ পেমেন্ট",
        "refund_request": "ফেরত অনুরোধ",
        "duplicate_payment": "ডুপ্লিকেট পেমেন্ট",
        "merchant_settlement_delay": "মার্চেন্ট সেটেলমেন্ট বিলম্ব",
        "agent_cash_in_issue": "এজেন্ট ক্যাশ-ইন সমস্যা",
        "phishing_or_social_engineering": "ফিশিং / সোশ্যাল ইঞ্জিনিয়ারিং",
        "other": "অন্যান্য (শ্রেণীবদ্ধ নয়)",
    }
    table = bn if lang == "bn" else en
    return table.get(case_type, case_type)


def _verdict_label(verdict: str, lang: str) -> str:
    en = {"consistent": "consistent", "inconsistent": "inconsistent", "insufficient_data": "insufficient data"}
    bn = {"consistent": "সামঞ্জস্যপূর্ণ", "inconsistent": "অসামঞ্জস্যপূর্ণ", "insufficient_data": "অপর্যাপ্ত তথ্য"}
    return (bn if lang == "bn" else en).get(verdict, verdict)


def _severity_label(severity: str, lang: str) -> str:
    en = {"low": "low", "medium": "medium", "high": "high", "critical": "critical"}
    bn = {"low": "নিম্ন", "medium": "মাঝারি", "high": "উচ্চ", "critical": "অত্যন্ত গুরুতর"}
    return (bn if lang == "bn" else en).get(severity, severity)


def _department_label(department: str, lang: str) -> str:
    en = {
        "customer_support": "Customer Support",
        "dispute_resolution": "Dispute Resolution",
        "payments_ops": "Payments Ops",
        "merchant_operations": "Merchant Operations",
        "agent_operations": "Agent Operations",
        "fraud_risk": "Fraud Risk",
    }
    bn = {
        "customer_support": "কাস্টমার সাপোর্ট",
        "dispute_resolution": "ডিসপিউট রিজলিউশন",
        "payments_ops": "পেমেন্টস অপারেশন্স",
        "merchant_operations": "মার্চেন্ট অপারেশন্স",
        "agent_operations": "এজেন্ট অপারেশন্স",
        "fraud_risk": "ফ্রড রিস্ক",
    }
    return (bn if lang == "bn" else en).get(department, department)


__all__ = ["TemplateReply", "render_safe_only", "render_templates"]
