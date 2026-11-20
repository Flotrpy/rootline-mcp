# Daily small-feature routine

Once a day, add one small, finished feature to rootline-mcp.

## Where the code lives

The Flotrpy/rootline-mcp repository on GitHub. Each run works on a new branch named
`daily/<Eastern date>` and opens a pull request into `main` for Affan to merge. Never push
to `main` or merge. Commits carry Affan (flotrpy) as the only author, with no AI
attribution lines.

## Each run

1. Set up: `uv venv && uv pip install -e ".[dev]"` in the folder.
2. Run `uv run pytest -q` first. If it already fails, fix that and stop there for the day.
3. Take the first unchecked item in the backlog below. Keep it to about an hour of work.
4. Build it as README "Adding a feature" says: logic in `tools.py`, registration in
   `server.py`, tests against the mock data, tool list in `tests/test_http.py`, a README
   table row and a CHANGELOG line dated in Eastern time.
5. Run `uv run pytest -q` and `uv run ruff check src tests && uv run ruff format --check src tests`.
   If either fails and you can't fix it, undo the day's change entirely.
6. Tick the item here with the date, then report in one or two sentences what was added.

Never deploy, push, create repositories, set secrets or change anything in Affan's
accounts without Affan's yes. Never ask for, copy or write an API key. Nothing paid.

## Backlog

- [x] (2026-11-18) `rootline_condition_effect(model)`: combined evidence vs the best single signal, paired, with McNemar (the paper's RQ1 check).
- [ ] `rootline_layer_gap(model)`: layer accuracy on network vs application faults with Fisher's exact test (RQ2).
- [ ] `link_rule="either_end"` option in score_summary: accept either service of a network link (the README's exploratory rescoring).
- [ ] `rootline_hardest_snapshots`: rank snapshots by how many model × condition runs got them right.
- [ ] `rootline_snapshot_brief(incident)`: one snapshot's fault and group, and which models and conditions got it right.
- [ ] `rootline_calibration(model)`: confidence bins vs accuracy and expected calibration error.
- [ ] `rootline_blame_map(model)`: for each fault, the services and categories the model blamed most often.
- [ ] `rootline_cost_speed`: prompt and answer tokens, latency and free-quota use per model and condition.
- [ ] `rootline_actions(model)`: remediation actions chosen, and how often the action fits the true layer.
- [ ] `rootline_perturbation_effect(model)`: p0 vs p1 (anonymised names), paired (RQ3 groundwork).
- [ ] `format="markdown"` option on score_summary and compare_models for pasting into the paper or README.
- [ ] MCP resources: the fault catalog and the latest results table; an MCP prompt "weekly benchmark report".
- [ ] Partial runs on GitHub: prepare (don't push) a `--conditions/--incidents` patch for `scripts/score_daily.py` plus the matching workflow inputs.
- [ ] `/status` page: last data refresh time, queue progress and version, for a quick look in a browser.
- [ ] Cache warm-up on start, so the first call after Render wakes is faster.
