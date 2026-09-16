"""
routers/mock.py
===============
Contract sandbox for the LLM team.

Serves frozen example payloads produced by the real fusion engine
(scripts/generate_contract_payloads.py). Loads no ML models, so it answers
instantly and works even while the registry is still warming up.

Why this exists: the live text endpoint can only ever produce single-signal
payloads. It never shows a face-vs-voice conflict, never shows confidence
dragged down by disagreement, never shows the empty transcript you get from
silence. Building against only those easy cases means breaking the first time a
real multimodal payload arrives. These examples cover the whole range.

    GET  /mock/scenarios          list every example with a one-line summary
    GET  /mock/payload/{name}     fetch one payload, byte-identical to the real thing
    POST /mock/emit/{name}        push one payload to LLM_ENDPOINT_URL,
                                  so the receiving end can be tested too
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import httpx
from fastapi import APIRouter, HTTPException

from src.core.config import LLM_ENDPOINT_URL

router = APIRouter()

PAYLOAD_DIR = os.path.join(PROJECT_ROOT, "contract", "payloads")


# ── Loading ───────────────────────────────────────────────────────────────────

def _read(name: str) -> dict:
    """
    Reads one payload file. The name is validated against the index rather than
    being used to build a path directly, so a crafted name cannot escape the
    payload directory.
    """
    path = os.path.join(PAYLOAD_DIR, f"{name}.json")
    if not os.path.isfile(path):
        raise HTTPException(
            status_code=404,
            detail=f"No such scenario '{name}'. See GET /mock/scenarios.",
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _index() -> dict:
    index_path = os.path.join(PAYLOAD_DIR, "index.json")
    if not os.path.isfile(index_path):
        raise HTTPException(
            status_code=503,
            detail=("Contract payloads have not been generated. "
                    "Run: python scripts/generate_contract_payloads.py"),
        )
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


def _known_names() -> set:
    return {s["name"] for s in _index().get("scenarios", [])}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/scenarios",
    summary="List example payloads",
    description="Every available example with the situation it represents.",
)
async def list_scenarios():
    return _index()


@router.get(
    "/payload/{name}",
    summary="Fetch one example payload",
    description=(
        "Returns a payload identical in shape to what /analyze/* emits. "
        "Use these to build and test your handling without running any models."
    ),
)
async def get_payload(name: str):
    if name not in _known_names():
        raise HTTPException(
            status_code=404,
            detail=f"No such scenario '{name}'. See GET /mock/scenarios.",
        )
    return _read(name)


@router.post(
    "/emit/{name}",
    summary="Push an example payload to the configured LLM endpoint",
    description=(
        "Sends the chosen example to LLM_ENDPOINT_URL exactly as the live "
        "pipeline would, so the receiving service can be tested end to end."
    ),
)
async def emit_payload(name: str):
    if name not in _known_names():
        raise HTTPException(
            status_code=404,
            detail=f"No such scenario '{name}'. See GET /mock/scenarios.",
        )

    if not LLM_ENDPOINT_URL or LLM_ENDPOINT_URL == "http://placeholder.url/endpoint":
        raise HTTPException(
            status_code=503,
            detail=("LLM_ENDPOINT_URL is not configured, so there is nowhere to "
                    "send this. Set it in .env first."),
        )

    payload = _read(name)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                LLM_ENDPOINT_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"{LLM_ENDPOINT_URL} did not respond within 10 seconds.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach {LLM_ENDPOINT_URL}: {exc}",
        )

    return {
        "scenario":       name,
        "delivered_to":   LLM_ENDPOINT_URL,
        "response_status": response.status_code,
        "response_body":  response.text[:2000],
    }
