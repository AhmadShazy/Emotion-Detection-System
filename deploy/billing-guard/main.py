"""
deploy/billing-guard/main.py
============================
A Cloud Function that DISABLES BILLING on this project when spending crosses a
hard limit.

Read this before deploying it
-----------------------------
This is a kill switch, not a throttle. When it fires it detaches the billing
account from the project, and every billable service stops -- Cloud Run returns
errors, Artifact Registry becomes unreadable, the deployed demo goes OFFLINE.
Nothing restarts on its own; billing has to be re-attached by hand in the
console.

That is the intended behaviour. It exists so a runaway cost cannot quietly
accumulate on a student account. But it means the failure mode is an outage,
which is worth knowing if a demo is scheduled.

Google deprecated real spending caps in 2020. A budget on its own only sends
email. This function is the standard pattern for turning a budget notification
into an actual stop.

How the pieces fit
------------------
    Cloud Billing Budget  ->  Pub/Sub topic  ->  this function  ->  billing off

The budget publishes a message every time it re-evaluates -- roughly several
times a day, NOT only when a threshold is crossed. So this function runs often
and must decide for itself whether to act. It does that by comparing the
reported cost against KILL_AMOUNT, and does nothing the vast majority of times
it is invoked.

Required IAM
------------
The function's service account needs BOTH of these, on two different
resources:

    roles/billing.user            on the BILLING ACCOUNT
    roles/billing.projectManager  on the PROJECT

Both are required to unlink a project from a billing account -- neither one
alone is enough, and gcloud rejects roles/billing.projectManager if you try to
bind it at the billing-account scope (it is a project-level role, not a
billing-account-level one). Without the full pair the disable call fails with
403 and the function logs an error rather than stopping anything -- which is
the worst of both worlds, so verify both bindings after deploying.
"""

import base64
import json
import os

import google.auth
from googleapiclient import discovery


# The spend at which billing is cut, in the budget's currency. Deliberately an
# environment variable rather than a constant: the number lives with the
# deployment, and changing it must not need a code edit.
KILL_AMOUNT = float(os.environ.get("KILL_AMOUNT", "6.0"))


def _resolve_project_id() -> str:
    """
    Find the project this function is running in.

    GCP_PROJECT and GOOGLE_CLOUD_PROJECT are gen1-only: Cloud Run (which is
    what gen2 functions run on) does not set either one automatically. Its
    reserved env vars are FUNCTION_TARGET, FUNCTION_SIGNATURE_TYPE, K_SERVICE,
    K_REVISION and PORT -- none of which name the project. Checking only the
    two gen1 vars, as an earlier version of this function did, silently
    returns an empty PROJECT_ID on gen2 and the function then refuses to act
    at all ("project id unavailable - cannot act") even when the kill
    threshold is legitimately crossed.

    google.auth.default() is the correct fallback: on Compute Engine, Cloud
    Run, or App Engine flexible/standard(2nd gen) -- which covers this
    function's actual runtime -- it resolves the project from the metadata
    service without needing any env var at all.
    """
    env_project = os.environ.get("GCP_PROJECT") or os.environ.get(
        "GOOGLE_CLOUD_PROJECT", ""
    )
    if env_project:
        return env_project

    try:
        _, detected = google.auth.default()
        return detected or ""
    except Exception as exc:
        print(f"[billing-guard] could not resolve project id via metadata service: {exc}")
        return ""


# projects/<id>.
PROJECT_ID = _resolve_project_id()
PROJECT_NAME = f"projects/{PROJECT_ID}"

# Set DRY_RUN=true to log the decision without touching billing. Worth doing on
# the first deploy: trigger the topic by hand and confirm the function reasons
# correctly before letting it hold a live kill switch.
DRY_RUN = os.environ.get("DRY_RUN", "false").strip().lower() in {"1", "true", "yes"}

_billing = discovery.build(
    "cloudbilling", "v1", cache_discovery=False
).projects()


def _billing_is_enabled() -> bool:
    """
    True when the project currently has a billing account attached.

    A failure here is reported as False rather than raised, because a broken
    read must not cause a disable attempt on a project that may already be
    detached.
    """
    try:
        info = _billing.getBillingInfo(name=PROJECT_NAME).execute()
        return bool(info.get("billingEnabled", False))
    except Exception as exc:
        print(f"[billing-guard] could not read billing state: {exc}")
        return False


def _disable_billing() -> None:
    """Detaches the billing account. This is the irreversible-ish part."""
    body = {"billingAccountName": ""}  # empty string means "none"
    result = _billing.updateBillingInfo(name=PROJECT_NAME, body=body).execute()
    print(f"[billing-guard] BILLING DISABLED for {PROJECT_NAME}: {result}")


def handle_budget_notification(event, context):  # noqa: ARG001 - CF signature
    """
    Pub/Sub entry point.

    The budget message carries, among other fields:
        costAmount      spend so far in the budget period
        budgetAmount    the configured budget
        currencyCode
        budgetDisplayName
        alertThresholdExceeded   present only once a threshold is crossed

    Deliberately keyed on costAmount rather than alertThresholdExceeded: the
    threshold field describes the BUDGET's own alert rules, which someone may
    edit in the console later. Comparing the actual spend to KILL_AMOUNT keeps
    the stop point defined here, where it is reviewable in version control.
    """
    try:
        raw = base64.b64decode(event["data"]).decode("utf-8")
        payload = json.loads(raw)
    except Exception as exc:
        print(f"[billing-guard] unreadable notification, ignoring: {exc}")
        return "bad payload"

    cost = float(payload.get("costAmount", 0.0))
    budget = float(payload.get("budgetAmount", 0.0))
    currency = payload.get("currencyCode", "")
    name = payload.get("budgetDisplayName", "<unnamed>")

    print(
        f"[billing-guard] budget={name!r} cost={cost} of {budget} {currency} "
        f"kill_at={KILL_AMOUNT} dry_run={DRY_RUN}"
    )

    if cost < KILL_AMOUNT:
        # The common case by a wide margin. Logged at low volume so the
        # function's own logs stay cheap and readable.
        return f"under limit ({cost} < {KILL_AMOUNT})"

    if not PROJECT_ID:
        print("[billing-guard] project id unavailable - cannot act")
        return "no project id"

    if not _billing_is_enabled():
        print("[billing-guard] billing already disabled, nothing to do")
        return "already disabled"

    if DRY_RUN:
        print(
            f"[billing-guard] DRY RUN - would disable billing on {PROJECT_NAME} "
            f"(cost {cost} >= {KILL_AMOUNT})"
        )
        return "dry run"

    print(
        f"[billing-guard] cost {cost} {currency} reached the {KILL_AMOUNT} "
        f"limit - disabling billing on {PROJECT_NAME}"
    )
    _disable_billing()
    return "billing disabled"
