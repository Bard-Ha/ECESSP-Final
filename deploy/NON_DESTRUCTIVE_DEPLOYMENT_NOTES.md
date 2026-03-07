# Non-Destructive Deployment Notes

No project files were deleted.

The deployment optimization strategy for ECESSP is:

- keep the full research repository locally,
- exclude large non-runtime assets from deployment packaging,
- deploy only the runtime subset needed by the frontend and backend services.

## What changed

The backend Docker context was reduced through:

- `ecessp-ml/.dockerignore`

This excludes bulk datasets from container builds while keeping the specific processed files used by the live runtime:

- `data/processed/material_catalog.csv`
- `data/processed/batteries_parsed.csv`
- `data/processed/batteries_parsed_curated.csv`
- `data/processed/batteries_ml_curated.csv`

## What did not change

- raw datasets were not deleted
- processed datasets were not deleted
- reports were not deleted
- graphs were not deleted
- checkpoints were not deleted

This is a deployment-only optimization, not a repository cleanup.
