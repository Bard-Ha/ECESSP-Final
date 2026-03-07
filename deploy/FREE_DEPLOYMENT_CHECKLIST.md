# ECESSP Free Deployment Checklist

This is the lowest-cost production split for this project:

- `ecessp.online` on `Vercel`
- `api.ecessp.online` on `Oracle Cloud Always Free`
- optional `Supabase Postgres` for frontend data

## 1. Frontend on Vercel

Deploy the `ecessp-frontend` directory as the Vercel project root.

### Required Vercel environment variables

- `VITE_ECESSP_API_BASE=https://api.ecessp.online/api`

### Optional Vercel environment variables

- `DATABASE_URL=<supabase postgres url>`

## 2. Backend on Oracle Always Free

Deploy `ecessp-ml` to an OCI VM or deploy the full repo and run only the backend service there.

### GitHub push boundary

Do not rely on a normal GitHub push for the full ML runtime assets.

This repository contains local data files larger than GitHub's 100 MB file limit, so the practical split is:

- push code and deploy configuration to GitHub
- clone that code-only repo on the OCI VM
- upload the required runtime assets directly to the VM with `scp`

### Runtime assets to upload directly to the OCI VM

Upload these local files after cloning the repo on the server:

- `ecessp-ml/data/processed/material_catalog.csv`
- `ecessp-ml/data/processed/batteries_parsed.csv`
- `ecessp-ml/data/processed/batteries_parsed_curated.csv`
- `ecessp-ml/data/processed/batteries_ml_curated.csv`
- `ecessp-ml/graphs/masked_battery_graph_normalized_v2.pt`
- `ecessp-ml/graphs/battery_hetero_graph_v1.pt`
- `ecessp-ml/reports/models/`
- `ecessp-ml/reports/training_summary.json`
- `ecessp-ml/reports/final_family_ensemble_manifest.json`
- `ecessp-ml/reports/three_model_ensemble_manifest.json`
- `ecessp-ml/reports/active_learning_queue.jsonl`

Example flow:

```powershell
# On the OCI VM after git clone
mkdir -p ~/ECESSP/ecessp-ml/data/processed
mkdir -p ~/ECESSP/ecessp-ml/graphs
mkdir -p ~/ECESSP/ecessp-ml/reports/models

# From your local machine
scp "ecessp-ml/data/processed/material_catalog.csv" ubuntu@<oci-ip>:~/ECESSP/ecessp-ml/data/processed/
scp "ecessp-ml/data/processed/batteries_parsed.csv" ubuntu@<oci-ip>:~/ECESSP/ecessp-ml/data/processed/
scp "ecessp-ml/data/processed/batteries_parsed_curated.csv" ubuntu@<oci-ip>:~/ECESSP/ecessp-ml/data/processed/
scp "ecessp-ml/data/processed/batteries_ml_curated.csv" ubuntu@<oci-ip>:~/ECESSP/ecessp-ml/data/processed/
scp "ecessp-ml/graphs/masked_battery_graph_normalized_v2.pt" ubuntu@<oci-ip>:~/ECESSP/ecessp-ml/graphs/
scp "ecessp-ml/graphs/battery_hetero_graph_v1.pt" ubuntu@<oci-ip>:~/ECESSP/ecessp-ml/graphs/
scp "ecessp-ml/reports/training_summary.json" ubuntu@<oci-ip>:~/ECESSP/ecessp-ml/reports/
scp "ecessp-ml/reports/final_family_ensemble_manifest.json" ubuntu@<oci-ip>:~/ECESSP/ecessp-ml/reports/
scp "ecessp-ml/reports/three_model_ensemble_manifest.json" ubuntu@<oci-ip>:~/ECESSP/ecessp-ml/reports/
scp "ecessp-ml/reports/active_learning_queue.jsonl" ubuntu@<oci-ip>:~/ECESSP/ecessp-ml/reports/
scp -r "ecessp-ml/reports/models" ubuntu@<oci-ip>:~/ECESSP/ecessp-ml/reports/
```

### Required backend environment variables

- `PORT=8000`
- `USE_GPU=0`
- `ECESSP_ALLOWED_ORIGINS=https://ecessp.online,https://www.ecessp.online`
- `REQUIRE_API_KEY=0`

### Recommended backend environment variables

- `RATE_LIMIT_ENABLED=1`
- `RATE_LIMIT_REQUESTS=120`
- `RATE_LIMIT_WINDOW_SEC=60`
- `REQUEST_TIMEOUT_SEC=60`

### Domain and TLS

Configure a reverse proxy so that:

- `api.ecessp.online` -> `127.0.0.1:8000`

## 3. DNS

Create these DNS records:

- `A` for `ecessp.online` -> Vercel
- `CNAME` or Vercel-managed record for `www.ecessp.online`
- `A` for `api.ecessp.online` -> OCI public IP

## 4. Authentication Note

If the frontend is deployed as a static Vercel site and calls the backend directly, do not require a secret backend API key from the browser.

Use one of these approaches:

- keep the backend public with rate limiting and CORS restrictions
- add real user auth later with Supabase Auth and backend token verification

## 5. Minimum Working Production State

The fastest free production setup is:

- Vercel static frontend
- OCI FastAPI backend
- no frontend proxy server
- backend protected by CORS + rate limiting

This avoids trying to force the long-running ML service into Vercel serverless.
