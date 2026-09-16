# Billing guard

Two layers, set up once, so a student account cannot quietly accumulate cost:

| Layer | Threshold | What it does |
|---|---|---|
| **Warning budget** | $1 and $2 | Emails you. Changes nothing. |
| **Kill switch** | $6 | Disables billing on the project. Everything stops. |

Expected steady-state spend for this project is around **$0.50/month** (Artifact
Registry storage for the ~5 GB image). Cloud Run should stay inside the Always
Free tier. So the warning should be rare and the kill switch should never fire
— it is there for the case where an assumption turns out to be wrong.

## Read this first

The kill switch is a **kill switch, not a throttle**. When it fires, billing is
detached from the project and the deployed app goes offline until you re-attach
it by hand. That is deliberate — an outage is the safe failure for a student
account — but do not deploy this the morning of a demo without understanding it.

Google removed real spending caps in 2020. A budget alone only sends email.
This function is the standard pattern for turning a budget into an actual stop.

## Why Cloud Shell

`gcloud` is not installed locally, and the development machine's connection has
been measured at 6–76 kB/s, which makes a ~150 MB SDK download impractical.
Cloud Shell has `gcloud` and `docker` pre-installed and already authenticated,
runs on Google's network, and is free (50 hours/week). Open it with the `>_`
icon in the top right of the Cloud Console.

Everything below is run there.

---

## 1. Set your project

```bash
gcloud config set project YOUR_PROJECT_ID
export PROJECT_ID=$(gcloud config get-value project)
export BILLING_ACCOUNT=$(gcloud billing projects describe "$PROJECT_ID" \
  --format="value(billingAccountName)" | sed 's|billingAccounts/||')

echo "project: $PROJECT_ID"
echo "billing: $BILLING_ACCOUNT"
```

Both must print a value. An empty billing account means the project is not
linked to one yet, and nothing below will work.

## 2. Enable the APIs

```bash
gcloud services enable \
  cloudbilling.googleapis.com \
  cloudfunctions.googleapis.com \
  pubsub.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com
```

`cloudbudgets.googleapis.com` is not a real, separately-enablable service --
`gcloud services enable` rejects it with `SERVICE_CONFIG_NOT_FOUND_OR_PERMISSION_DENIED`.
Budget creation in Step 6 rides on `cloudbilling.googleapis.com`, which is
already in the list above.

## 3. Create the Pub/Sub topic the budget publishes to

```bash
gcloud pubsub topics create billing-alerts
```

## 4. Deploy the kill switch

From the repository root (clone it in Cloud Shell first — it is a few MB):

```bash
gcloud functions deploy billing-guard \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=deploy/billing-guard \
  --entry-point=handle_budget_notification \
  --trigger-topic=billing-alerts \
  --set-env-vars=KILL_AMOUNT=6,DRY_RUN=true \
  --memory=256Mi
```

**Note `DRY_RUN=true`.** The first deploy deliberately cannot disable anything —
it only logs what it *would* do. Step 7 turns it off once you have seen it
reason correctly.

## 5. Grant it permission to disable billing

This is the step people miss, and it needs **two** bindings on **two different
resources** -- one on the billing account, one on the project. Neither alone is
enough, and `gcloud` rejects `roles/billing.projectManager` if you try to bind
it at the billing-account scope, since it is a project-level role.

```bash
SA=$(gcloud functions describe billing-guard --gen2 --region=us-central1 \
  --format="value(serviceConfig.serviceAccountEmail)")
echo "service account: $SA"

# Billing-account-level role: lets the SA unlink projects from this billing account
gcloud billing accounts add-iam-policy-binding "$BILLING_ACCOUNT" \
  --member="serviceAccount:$SA" \
  --role="roles/billing.user"

# Project-level role: lets the SA actually change this project's billing link
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA" \
  --role="roles/billing.projectManager"
```

Without both, the disable call fails with 403 at exactly the moment it
matters.

## 6. Create both budgets

**Warning budget — $2, email only:**

```bash
gcloud billing budgets create \
  --billing-account="$BILLING_ACCOUNT" \
  --display-name="warn-2usd" \
  --budget-amount=2USD \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=100
```

Thresholds at 50% and 100% of $2 give you alerts at **$1 and $2**.

**Kill budget — $6, wired to the function:**

```bash
gcloud billing budgets create \
  --billing-account="$BILLING_ACCOUNT" \
  --display-name="kill-6usd" \
  --budget-amount=6USD \
  --threshold-rule=percent=100 \
  --all-updates-rule-pubsub-topic="projects/$PROJECT_ID/topics/billing-alerts"
```

Budget emails go to the billing account's Billing Admins by default. Check that
address is one you actually read — Console → Billing → Budgets & alerts → edit
→ *Manage notifications*.

## 7. Test it, then arm it

Do not skip this. An untested kill switch is an assumption, not a safeguard.

Publish a fake notification claiming $10 of spend:

```bash
gcloud pubsub topics publish billing-alerts --message='{
  "budgetDisplayName": "kill-6usd",
  "costAmount": 10.0,
  "budgetAmount": 6.0,
  "currencyCode": "USD"
}'

# give it a few seconds, then read the logs
gcloud functions logs read billing-guard --gen2 --region=us-central1 --limit=20
```

You should see a line containing **`DRY RUN - would disable billing`**. That
proves the whole chain works: budget topic, function, decision logic, and the
permission check — without taking anything down.

Also confirm the quiet path. Publish a message with `"costAmount": 0.5` and the
logs should show `under limit`.

Once both behave correctly, arm it:

```bash
gcloud functions deploy billing-guard \
  --gen2 --region=us-central1 \
  --source=deploy/billing-guard \
  --entry-point=handle_budget_notification \
  --trigger-topic=billing-alerts \
  --set-env-vars=KILL_AMOUNT=6,DRY_RUN=false \
  --memory=256Mi
```

## If it ever fires

1. Console → Billing → **Account management** → re-link the billing account
2. Work out what consumed the money before re-deploying anything —
   Console → Billing → **Reports**, grouped by service
3. The most likely culprit by far is `--min-instances` left set on Cloud Run,
   which keeps a container warm around the clock and burns the free tier in
   about a day

## Cost of the guard itself

Nothing. Pub/Sub and Cloud Functions both have free tiers far larger than this
uses — the function runs a handful of times a day and does almost nothing each
time.
