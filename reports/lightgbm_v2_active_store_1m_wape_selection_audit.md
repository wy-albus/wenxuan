# LightGBM V2 Active-Store 1M WAPE Selection Audit

## Scope

- Dataset: `data/processed/model_dataset_monthly_active_store.parquet`
- Horizon: 1M only
- Model: LightGBM V2 Log-L2 (`log1p` target, `regression_l2`)
- Checkpoint: `models/checkpoints/active_store_experiment/lightgbm_v2_logl2_1m.txt`
- LightGBM version: 4.6.0
- Input features: 57
- Selection data: complete eligible Valid split on the original quantity scale
- Complete Valid rows per candidate: 18,563,573
- Candidate iterations evaluated: 14

## Selection Result

The candidate selector minimizes Valid WAPE first, then absolute total bias, then maximizes trend Macro-F1, and finally prefers fewer iterations. WAPE is compared strictly because no numerical closeness tolerance has been configured.

| Item | Value |
|---|---:|
| Selected iteration | 700 |
| Saved model trees | 700 |
| Valid WAPE | 117.2095% |
| Valid MAE | 0.155863 |
| Valid RMSE | 2.150879 |
| Valid true quantity | 2,468,546 |
| Valid predicted quantity | 1,788,878.99 |
| Valid total bias | -27.5331% |
| Valid trend Macro-F1 | 0.373646 |
| Checkpoint size | 5.81 MiB |

The sampled-validation L2 proxy stopped at 832 iterations. On complete Valid, iteration 700 had the lowest WAPE. Iteration 832 had WAPE 117.2146%, total bias -27.1225%, and trend Macro-F1 0.377208, so it was not selected under strict WAPE-first ordering.

## Verification

- All 14 candidates contain finite metrics and exactly 18,563,573 observations.
- Recomputing the documented ordering from checkpoint metadata selects iteration 700.
- The checkpoint reloads successfully and contains exactly 700 trees in the persisted 57-feature order.
- Active-store and candidate-selection tests: 43 passed.
- No 2M, two-stage, or MC formal training was resumed after the interrupted run.

## Status

The complete-Valid WAPE selection mechanism is operational and auditable for LightGBM V2 1M. The checkpoint remains under `models/checkpoints/` and has not been promoted over any existing final model.
