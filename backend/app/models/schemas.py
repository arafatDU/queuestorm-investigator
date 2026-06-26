"""Pydantic request and response schemas for the QueueStorm Investigator API.

All enum values are bound to the exact strings defined in ``AGENT.md`` and the
sample case pack. The enums are declared as standard :class:`enum.Enum` (not
``str, Enum``) and serialized via Pydantic ``use_enum_values`` semantics so the
JSON output is always the raw string.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums (must match AGENT.md exactly)
# ---------------------------------------------------------------------------


class CaseType(str, Enum):
    """Allowed values for :attr:`AnalyzeTicketResponse.case_type`."""

    WRONG_TRANSFER = "wrong_transfer"
    PAYMENT_FAILED = "payment_failed"
    REFUND_REQUEST = "refund_request"
    DUPLICATE_PAYMENT = "duplicate_payment"
    MERCHANT_SETTLEMENT_DELAY = "merchant_settlement_delay"
    AGENT_CASH_IN_ISSUE = "agent_cash_in_issue"
    PHISHING_OR_SOCIAL_ENGINEERING = "phishing_or_social_engineering"
    OTHER = "other"


class Department(str, Enum):
    """Allowed values for :attr:`AnalyzeTicketResponse.department`."""

    CUSTOMER_SUPPORT = "customer_support"
    DISPUTE_RESOLUTION = "dispute_resolution"
    PAYMENTS_OPS = "payments_ops"
    MERCHANT_OPERATIONS = "merchant_operations"
    AGENT_OPERATIONS = "agent_operations"
    FRAUD_RISK = "fraud_risk"


class EvidenceVerdict(str, Enum):
    """Allowed values for :attr:`AnalyzeTicketResponse.evidence_verdict`."""

    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    INSUFFICIENT_DATA = "insufficient_data"


class Severity(str, Enum):
    """Allowed values for :attr:`AnalyzeTicketResponse.severity`."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# --- Optional enums from the sample case pack ------------------------------


class Language(str, Enum):
    """Optional ``language`` hint on incoming tickets."""

    EN = "en"
    BN = "bn"
    MIXED = "mixed"


class Channel(str, Enum):
    """Optional ``channel`` hint describing where the ticket originated."""

    IN_APP_CHAT = "in_app_chat"
    CALL_CENTER = "call_center"
    EMAIL = "email"
    MERCHANT_PORTAL = "merchant_portal"
    FIELD_AGENT = "field_agent"


class UserType(str, Enum):
    """Optional ``user_type`` hint describing the complainant."""

    CUSTOMER = "customer"
    MERCHANT = "merchant"
    AGENT = "agent"
    UNKNOWN = "unknown"


class TransactionType(str, Enum):
    """Transaction type from the supplied ``transaction_history``."""

    TRANSFER = "transfer"
    PAYMENT = "payment"
    CASH_IN = "cash_in"
    CASH_OUT = "cash_out"
    SETTLEMENT = "settlement"
    REFUND = "refund"


class TransactionStatus(str, Enum):
    """Transaction status from the supplied ``transaction_history``."""

    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"
    REVERSED = "reversed"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class TransactionRecord(BaseModel):
    """A single transaction supplied as part of the complaint context."""

    model_config = ConfigDict(extra="ignore")

    transaction_id: str = Field(..., description="Unique transaction identifier.")
    timestamp: Optional[datetime] = Field(
        default=None, description="ISO-8601 timestamp of the transaction."
    )
    type: Optional[TransactionType] = Field(
        default=None, description="Transaction type (transfer, payment, etc.)."
    )
    amount: Optional[float] = Field(
        default=None, ge=0, description="Transaction amount in the local currency."
    )
    counterparty: Optional[str] = Field(
        default=None, description="Counterparty identifier (phone, agent, merchant)."
    )
    status: Optional[TransactionStatus] = Field(
        default=None, description="Final transaction status."
    )


class AnalyzeTicketRequest(BaseModel):
    """Inbound payload for ``POST /analyze-ticket``."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    ticket_id: str = Field(..., description="Unique ticket identifier.")
    complaint: str = Field(
        ..., description="Raw complaint text written by the user."
    )

    # --- Optional hints ----------------------------------------------------
    language: Optional[Language] = Field(
        default=None, description="Detected or declared complaint language."
    )
    channel: Optional[Channel] = Field(
        default=None, description="Channel where the ticket originated."
    )
    user_type: Optional[UserType] = Field(
        default=None, description="Type of user filing the complaint."
    )
    campaign_context: Optional[str] = Field(
        default=None, description="Optional campaign or promotion identifier."
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Free-form metadata bag."
    )

    # --- Transaction context ----------------------------------------------
    transaction_history: list[TransactionRecord] = Field(
        default_factory=list,
        description="Recent transactions relevant to the complaint.",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AnalyzeTicketResponse(BaseModel):
    """Outbound payload for ``POST /analyze-ticket``."""

    model_config = ConfigDict(extra="ignore")

    ticket_id: str = Field(..., description="Echoes the inbound ticket identifier.")
    relevant_transaction_id: Optional[str] = Field(
        default=None,
        description="ID of the transaction matched to the complaint, or null.",
    )
    evidence_verdict: EvidenceVerdict = Field(
        ..., description="Result of cross-checking the complaint against history."
    )
    case_type: CaseType = Field(..., description="Classified complaint category.")
    severity: Severity = Field(..., description="Operational severity tier.")
    department: Department = Field(..., description="Team that should own the ticket.")
    agent_summary: str = Field(
        ..., min_length=1, description="LLM-generated summary for the human agent."
    )
    recommended_next_action: str = Field(
        ..., min_length=1, description="LLM-generated suggested next step."
    )
    customer_reply: str = Field(
        ..., min_length=1, description="LLM-generated (and sanitized) customer reply."
    )
    human_review_required: bool = Field(
        ..., description="True when the ticket must be escalated to a human."
    )

    # --- Optional enrichment ----------------------------------------------
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional model confidence score in [0, 1].",
    )
    reason_codes: Optional[list[str]] = Field(
        default=None, description="Optional short tags describing key decisions."
    )


class HealthResponse(BaseModel):
    """Outbound payload for ``GET /health``."""

    status: str = Field(default="ok", description="Service status indicator.")


__all__ = [
    "AnalyzeTicketRequest",
    "AnalyzeTicketResponse",
    "CaseType",
    "Channel",
    "Department",
    "EvidenceVerdict",
    "HealthResponse",
    "Language",
    "Severity",
    "TransactionRecord",
    "TransactionStatus",
    "TransactionType",
    "UserType",
]