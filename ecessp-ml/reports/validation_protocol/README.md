# ECESSP DFT + Experiment Validation Protocol

This package quantifies whether ECESSP predictions are externally reliable enough for stronger claims.

## Files

- `dft_experiment_validation_protocol_v1.json`
  - Machine-readable gates, acceptance thresholds, and confidence-index weights.
- `templates/predictions_manifest_template.csv`
  - Model outputs to validate.
- `templates/dft_results_template.csv`
  - DFT outcomes for the same candidate IDs.
- `templates/experiment_results_template.csv`
  - Experimental outcomes for the same candidate IDs.
- `tools/run_dft_experiment_validation.py`
  - Computes pass rates, error metrics, uncertainty calibration, confidence index, and decision.

## Required Candidate ID Contract

Use a stable `candidate_id` across:

- prediction manifest
- DFT result row
- experiment result row

Rows without matching `candidate_id` in the prediction manifest are skipped.

## Validation Ladder (Execution Order)

1. Freeze model/checkpoint and export a prediction manifest for top candidates.
2. Run DFT validation on the same candidate IDs.
3. Run lab experiments on a DFT-passing subset.
4. Run the protocol evaluator and inspect the generated report.
5. Accept or reject scale-up claims using `decision.ready_for_scale_up`.

## Run

From `ecessp-ml`:

```powershell
python tools/run_dft_experiment_validation.py `
  --predictions-csv reports/validation_protocol/templates/predictions_manifest_template.csv `
  --dft-csv reports/validation_protocol/templates/dft_results_template.csv `
  --experiment-csv reports/validation_protocol/templates/experiment_results_template.csv `
  --output-json reports/validation_protocol/dft_experiment_validation_report.json
```

## Report Outputs

The generated JSON includes:

- `stage_metrics.dft` and `stage_metrics.experiment`:
  - attempted/pass counts
  - pass rates
  - voltage MAE
  - per-candidate pass/fail reasons
- `calibration`:
  - Pearson correlation between predicted uncertainty and observed absolute voltage error
- `acceptance_gates`:
  - explicit gate-by-gate pass/fail
- `confidence`:
  - `confidence_index_0_to_100`
  - confidence tier (`high`, `medium`, `low`)
- `decision.ready_for_scale_up`:
  - final go/no-go indicator

## Policy Notes

- This protocol is insertion-type only.
- Passing this protocol increases trust; it does not replace full safety qualification or manufacturing validation.
- If gates fail, route candidates into active learning and retrain/recalibrate before making reliability claims.
