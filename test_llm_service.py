"""Smoke test for LLMOrchestrator — no real API call."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.services.investigator import TransactionInvestigator
from app.services.llm_service import (
    LLMOrchestrator,
    _extract_text_from_gemini_response,
    _parse_generated_json,
    _strip_code_fences,
)


def main() -> int:
    inv = TransactionInvestigator()
    result = inv.investigate(
        complaint=(
            "I sent 5000 taka to a wrong number around 2pm today. "
            "The number was supposed to be 01712345678."
        ),
        transaction_history=[
            {
                "transaction_id": "TXN-9101",
                "timestamp": "2026-04-14T14:08:22Z",
                "type": "transfer",
                "amount": 5000,
                "counterparty": "+8801719876543",
                "status": "completed",
            }
        ],
    )
    print("Investigation:", result.as_dict())

    llm = LLMOrchestrator()
    prompt = llm._build_user_prompt(
        ticket_id="TKT-001",
        complaint="I sent 5000 taka to a wrong number around 2pm today.",
        language="en",
        user_type="customer",
        campaign_context="boishakh_bonanza_day_1",
        investigation=result,
    )
    print("\n--- USER PROMPT ---")
    print(prompt)

    body = llm._build_request_body(prompt, "en")
    print("\n--- REQUEST BODY KEYS ---")
    print(list(body.keys()))
    print("\n--- GENERATION CONFIG ---")
    print(body["generationConfig"])
    print("\n--- SAFETY SETTINGS ---")
    print(body["safetySettings"])
    print("\n--- SYSTEM PROMPT (first 500 chars) ---")
    print(body["systemInstruction"]["parts"][0]["text"][:500])

    # Test JSON parsing helpers
    raw_fenced = """
```json
{
  "agent_summary": "Customer reports a wrong-number transfer of 5000 BDT (TXN-9101).",
  "recommended_next_action": "Verify TXN-9101 and initiate dispute.",
  "customer_reply": "We have noted your concern about TXN-9101."
}
```
"""
    parsed = _parse_generated_json(raw_fenced)
    print("\n--- PARSED (fenced) ---")
    print(parsed)
    assert parsed["agent_summary"].startswith("Customer reports")

    raw_prose = "Sure! Here you go: {\"agent_summary\":\"x\",\"recommended_next_action\":\"y\",\"customer_reply\":\"z\"} -- cheers!"
    parsed2 = _parse_generated_json(raw_prose)
    print("\n--- PARSED (prose-wrapped) ---")
    print(parsed2)
    assert parsed2["agent_summary"] == "x"

    # Missing field → ValueError
    try:
        _parse_generated_json('{"agent_summary":"x","recommended_next_action":"y"}')
    except ValueError as e:
        print("\n--- MISSING FIELD DETECTED ---")
        print(str(e))
        assert "customer_reply" in str(e)

    # Gemini-shaped payload parsing
    fake_payload = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"text": '{"agent_summary":"a","recommended_next_action":"b","customer_reply":"c"}'}
                    ]
                },
            }
        ]
    }
    text = _extract_text_from_gemini_response(fake_payload)
    print("\n--- EXTRACTED FROM GEMINI ---")
    print(text)
    assert text is not None

    # Safety-blocked payload → None
    blocked = {
        "candidates": [
            {"finishReason": "SAFETY", "content": {"parts": []}}
        ]
    }
    text2 = _extract_text_from_gemini_response(blocked)
    print("Blocked payload returns:", text2)
    assert text2 is None

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())