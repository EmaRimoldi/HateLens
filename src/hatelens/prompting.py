"""Optional policy / moderation-definition conditioning (Phase 6)."""

from __future__ import annotations


def prepend_policy(text: str, policy_text: str | None) -> str:
    """Prepend a policy string when present; otherwise return ``text`` unchanged."""
    if not policy_text or not policy_text.strip():
        return text
    return f"[POLICY] {policy_text.strip()}\n[TEXT] {text}"
