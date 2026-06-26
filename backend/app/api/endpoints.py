"""HTTP route handlers for the QueueStorm Investigator API.

This module owns the single business endpoint ``POST /analyze-ticket``.
The pipeline is:

1. Validate the inbound payload (``AnalyzeTicketRequest``).
2. Run the deterministic :class:`TransactionInvestigator` to get
   ``case_type``, ``evidence_verdict``, ``severity``, ``department``,
   ``relevant_transaction_id``, ``human_review_required``, ``reason_codes``.
3. Call the asynchronous :class:`LLMOrchestrator` with the deterministic
   facts as ground truth. If the LLM call fails or times out, fall back to
   deterministic templates (the request still finishes well within the 30s
   SLA).
4. Sanitize the LLM-generated ``customer_reply`` and
   ``recommended_next_action`` through :class:`SafetyValidator`. If any
   rule violation is detected the unsafe text is overwritten with a
   hardcoded safe fallback.
5. OR the validator's ``should_escalate`` decision with the investigator's
   own ``human_review_required`` flag so safety-sensitive cases always
   escalate.
6. Build the typed :class:`AnalyzeTicketResponse` and return ``200``.

Global exception handlers (malformed JSON, Pydantic validation errors,
empty complaint, internal exceptions) are registered in :mod:`app.main`.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, status

from app.models.schemas import (
    AnalyzeTicketRequest,
    AnalyzeTicketResponse,
)
from app.services.investigator import TransactionInvestigator
from app.services.llm_service import LLMOrchestrator
from app.services.safety import SafetyValidator
from app.services.templates import render_safe_only, render_templates


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Router + collaborators
# ---------------------------------------------------------------------------


router = APIRouter(tags=["analyze"])

# Module-level singletons. The investigator is pure-Python and stateless; the
# validator is also stateless; the LLM orchestrator holds a settings
# reference but no connection pool until the first call.
_investigator = TransactionInvestigator()
_safety = SafetyValidator()
_llm = LLMOrchestrator()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/analyze-ticket",
    response_model=AnalyzeTicketResponse,
    status_code=status.HTTP_200_OK,
    summary="Investigate a customer complaint and return a routed response.",
)
async def analyze_ticket(payload: AnalyzeTicketRequest) -> AnalyzeTicketResponse:
    """Run the full investigator → LLM → safety pipeline.

    Always returns ``200`` with a valid response body when the request
    conforms to the schema. Validation errors are handled by the global
    exception handlers in ``app.main`` and converted to ``400`` / ``422``.
    """

    started = time.perf_counter()

    # --- 0. Semantic checks (per PS §4.1: 422 for invalid but well-formed input) --
    # The Pydantic schema intentionally allows empty strings so the request can
    # be parsed; semantically-empty inputs raise ValueError → global handler
    # maps them to 422 instead of 400.
    if not (payload.ticket_id or "").strip():
        raise ValueError("ticket_id must be a non-empty string.")
    if not (payload.complaint or "").strip():
        raise ValueError("complaint must be a non-empty string.")

    # --- 1. Deterministic investigation ---------------------------------
    investigation = _investigator.investigate_request(payload)

    # --- 2. Language hint + LLM call (with deterministic fallback) -------
    language_hint = (
        payload.language.value if payload.language is not None else None
    )

    generated: Optional[object] = None
    try:
        generated = await _llm.generate(
            ticket_id=payload.ticket_id,
            complaint=payload.complaint,
            language=language_hint or investigation.signals.language_hint,
            user_type=(
                payload.user_type.value if payload.user_type is not None else None
            ),
            campaign_context=payload.campaign_context,
            investigation=investigation,
        )
    except Exception as exc:  # noqa: BLE001 — defensive, must never 500
        logger.warning("LLMOrchestrator raised unexpectedly: %s", exc)
        generated = None

    if generated is not None:
        candidate_reply = generated.customer_reply  # type: ignore[attr-defined]
        candidate_next_action = generated.recommended_next_action  # type: ignore[attr-defined]
        candidate_summary = generated.agent_summary  # type: ignore[attr-defined]
    else:
        fallback = render_templates(investigation, language=language_hint)
        candidate_reply = fallback.customer_reply
        candidate_next_action = fallback.recommended_next_action
        candidate_summary = fallback.agent_summary

    # --- 3. Safety sanitizer over LLM output -----------------------------
    decision = _safety.sanitize(
        customer_reply=candidate_reply,
        recommended_next_action=candidate_next_action,
        complaint_language=language_hint,
    )

    # If the sanitizer overwrote the customer_reply, regenerate the
    # agent_summary / next_action from the safe-only template so the three
    # fields stay coherent.
    if decision.reply_was_overwritten or decision.next_action_was_overwritten:
        safe_only = render_safe_only(decision.language)
        candidate_summary = safe_only.agent_summary
        if decision.next_action_was_overwritten:
            candidate_next_action = safe_only.recommended_next_action
        if decision.reply_was_overwritten:
            candidate_reply = safe_only.customer_reply
    else:
        candidate_reply = decision.customer_reply
        candidate_next_action = decision.recommended_next_action

    # --- 4. human_review_required OR with the safety validator ----------
    escalate, _reasons = _safety.should_escalate(
        case_type=investigation.case_type,
        severity=investigation.severity,
        evidence_verdict=investigation.evidence_verdict,
        complaint_text=payload.complaint,
    )
    human_review_required = investigation.human_review_required or escalate

    # --- 5. Reason codes -------------------------------------------------
    reason_codes = list(investigation.reason_codes)
    if escalate and "safety_validator_escalation" not in reason_codes:
        reason_codes.append("safety_validator_escalation")

    # --- 6. Confidence (rough heuristic) --------------------------------
    confidence = _compute_confidence(investigation)

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "analyze-ticket: ticket_id=%s case=%s severity=%s verdict=%s "
        "review=%s llm=%s elapsed_ms=%.1f",
        payload.ticket_id,
        investigation.case_type.value,
        investigation.severity.value,
        investigation.evidence_verdict.value,
        human_review_required,
        "llm" if generated is not None else "fallback",
        elapsed_ms,
    )

    return AnalyzeTicketResponse(
        ticket_id=payload.ticket_id,
        relevant_transaction_id=investigation.relevant_transaction_id,
        evidence_verdict=investigation.evidence_verdict,
        case_type=investigation.case_type,
        severity=investigation.severity,
        department=investigation.department,
        agent_summary=candidate_summary,
        recommended_next_action=candidate_next_action,
        customer_reply=candidate_reply,
        human_review_required=human_review_required,
        confidence=confidence,
        reason_codes=reason_codes or None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_confidence(investigation) -> float:
    """Return a rough confidence score in [0, 1]."""

    score = 0.7
    if investigation.relevant_transaction_id:
        score += 0.15
    if investigation.evidence_verdict.value == "consistent":
        score += 0.1
    elif investigation.evidence_verdict.value == "inconsistent":
        score += 0.05
    # Vague complaints lower confidence
    if investigation.evidence_verdict.value == "insufficient_data":
        score -= 0.1
    return max(0.4, min(0.95, round(score, 2)))
