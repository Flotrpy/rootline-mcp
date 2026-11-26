# Changelog

One line per change, newest first. The daily feature routine adds to this.

## Unreleased

- 2026-11-26: `rootline_actions` shows which remediation actions a model picks and how often they act on the true fault's layer.

- 2026-11-25: `rootline_cost_speed` shows tokens and latency per model and condition, and how many runs a day each free tier allows.

- 2026-11-24: `rootline_blame_map` shows what a model blamed most often for each fault. Calibration bin labels now show exact edges (from review).

- 2026-11-23: `rootline_calibration` compares a model's stated confidence with its top-1 accuracy, with expected calibration error.

- 2026-11-22: `rootline_snapshot_brief` shows one snapshot's fault, group and every run's result. Readings now say "no statistically significant difference" instead of implying there is none (from review).

- 2026-11-21: `rootline_hardest_snapshots` ranks snapshots by how many runs got them right. Fixes from review: either-end scoring skips answers the harness couldn't parse; `rootline_layer_gap` counts hardware cases as network like the paper's RQ2, calls its p-value unadjusted, and gives no significance reading when conditions are pooled.

- 2026-11-20: `link_rule="either_end"` on `rootline_score_summary`: exploratory rescoring
  that accepts either end of a broken link for link faults.
- 2026-11-19: `rootline_layer_gap`: layer accuracy on network vs application faults with
  Fisher's exact test (RQ2).
- 2026-11-18: `rootline_condition_effect`: combined evidence vs each single signal,
  paired on the snapshots all four conditions share, with McNemar's exact test against
  the best single signal (RQ1).

## 0.1.0 (2026-11-18)

- First version: 9 tools (overview, list_models, start_run, run_status, list_failures,
  merge_results, score_summary, compare_models, show_cases), GitHub Actions and local run
  backends, Hugging Face and local data sources, mock mode, token-protected HTTP endpoint.
