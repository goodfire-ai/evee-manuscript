"""Thin client for the public EVEE API.

Helpers vendored and trimmed from goodfire-ai/evee-mcp/server.py. We keep the
raw responses rather than curating them — downstream scripts can slice.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

BASE_URL = "https://xix0d0o8le.execute-api.us-east-1.amazonaws.com"


def make_client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=timeout)


def evee_url(variant_id: str) -> str:
    return f"https://evee.goodfire.ai/#/variant/{variant_id}"


def get_variant(client: httpx.Client, variant_id: str) -> tuple[int, dict[str, Any] | None]:
    """GET /variants/{id}. Returns (status_code, json_or_None)."""
    resp = client.get(f"/variants/{variant_id}")
    if resp.status_code == 404:
        return 404, None
    resp.raise_for_status()
    return resp.status_code, resp.json()


def fetch_analysis(client: httpx.Client, variant_id: str) -> dict[str, Any]:
    """GET /variants/{id}/analysis.

    Returns one of:
      {"status": "complete", "result": {...}}
      {"status": "queued"|"processing", "retry_after": N}
      {"status": "not_found"}
    """
    resp = client.get(f"/variants/{variant_id}/analysis")
    if resp.status_code == 404:
        return {"status": "not_found"}
    resp.raise_for_status()
    return resp.json()


def wait_for_analysis(
    client: httpx.Client,
    variant_id: str,
    overall_timeout: float = 300.0,
    initial_poll: float = 5.0,
    max_poll: float = 30.0,
) -> dict[str, Any]:
    """Poll /analysis with exponential backoff until complete, not_found, or timeout.

    Returns the final analysis dict. On timeout, returns {"status": "timeout"}.
    """
    deadline = time.monotonic() + overall_timeout
    poll = initial_poll

    while True:
        analysis = fetch_analysis(client, variant_id)
        status = analysis.get("status")
        if status in ("complete", "not_found"):
            return analysis
        if time.monotonic() >= deadline:
            return {"status": "timeout", "last": analysis}
        retry_after = analysis.get("retry_after")
        sleep = float(retry_after) if retry_after else poll
        remaining = deadline - time.monotonic()
        sleep = min(sleep, remaining)
        if sleep > 0:
            time.sleep(sleep)
        poll = min(poll * 1.5, max_poll)


def extract_interpretation(response: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the interpretation out of a /variants/{id} response.

    Returns a curated dict with summary/mechanism/key_evidence/confidence, or
    None if no stored interpretation is present and status != ok.
    """
    pr = response.get("processed_result")
    if not isinstance(pr, dict) or pr.get("status") != "ok":
        return None
    return {
        "summary": pr.get("summary"),
        "mechanism": pr.get("mechanism"),
        "key_evidence": pr.get("key_evidence"),
        "confidence": pr.get("confidence"),
    }


def interpretation_from_analysis(analysis: dict[str, Any]) -> dict[str, Any] | None:
    """Pull interpretation out of a /analysis response."""
    if analysis.get("status") != "complete":
        return None
    r = analysis.get("result") or {}
    return {
        "summary": r.get("summary"),
        "mechanism": r.get("mechanism"),
        "key_evidence": r.get("key_evidence"),
        "confidence": r.get("confidence"),
    }
