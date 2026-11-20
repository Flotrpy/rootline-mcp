# Changelog

One line per change, newest first. The daily feature routine adds to this.

## Unreleased

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
