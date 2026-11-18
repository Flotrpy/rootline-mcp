# rootline-mcp

An MCP connector for [Rootline](https://github.com/Flotrpy/RootLine). It lets Claude
start scoring runs, watch the job queues, check and merge results, and dig into what the
models actually answered. Free to run: no paid service, no card.

## Tools

| Tool | What it does |
|---|---|
| `rootline_overview` | What the connector can see: result files, runs per model, queue progress, whether it can start runs. |
| `rootline_list_models` | Every model, where it runs (free cloud API or Ollama on your computer), free jobs per day, runs already scored, days left to finish. |
| `rootline_start_run` | Plans a run (jobs left, days at the free cap). With `confirm=true` it starts it on GitHub Actions or your computer. The queue resumes, so nothing is scored twice. |
| `rootline_run_status` | Done, pending and failed jobs per queue, percent done, days left, last activity in Eastern time, and links to the latest runs. |
| `rootline_list_failures` | Failed jobs grouped by error, with examples, and how many answers failed to parse. |
| `rootline_merge_results` | Checks the per-provider exports before merging: conflicting run ids, queue count vs export rows, missing snapshot × condition cells, unparsed answers. `write=true` saves `runs.parquet` to the private working dataset. |
| `rootline_score_summary` | Top-1 with a 95% bootstrap interval, top-3, layer, evidence validity, parse rate, confidence and healthy false alarms, grouped by model, condition, fault, fault group (application / network / hardware / healthy), perturbation or incident. |
| `rootline_compare_models` | Paired comparison on the same snapshots with McNemar's exact test and example snapshots where the two models differ. |
| `rootline_show_cases` | Real answers next to the true fault: wrong, right, unparsed, overconfident or false alarms on healthy snapshots. |

## How it works

```
Claude ──MCP──▶ rootline-mcp (Render free, or your computer)
                  │ reads                         │ starts
                  ▼                               ▼
     Hugging Face datasets               GitHub Actions (free for public repos)
     • Flotrpy/rootline (public)          └─ .github/workflows/rootline-score.yml
       runs/runs.parquet                       └─ scripts/score_daily.py --model …
     • Flotrpy/rootline-work (private)              resumes queue-<provider>.sqlite,
       queue-<provider>.sqlite                      scores one free-tier day,
       runs-<provider>.parquet  ◀───────────────── uploads queue + export
```

The connector never holds a model API key. The Groq key lives in the RootLine repo's
Actions secrets and is used only by the workflow, which runs the harness exactly as
`scripts/score_daily.py` already does. Ollama models can't run on GitHub's runners, so
they use the local backend on your computer.

## Try it with made-up data

```bash
uv run --with . rootline-mcp --mock --port 8000
# MCP endpoint: http://localhost:8000/mcp
```

Or for Claude Code / Claude Desktop over stdio:

```bash
claude mcp add rootline -- uv run --directory /path/to/rootline-mcp rootline-mcp --stdio --mock
```

## Use it on your own computer

Point it at your RootLine checkout. It reads `data/runs/*.parquet` and
`data/queue-*.sqlite`, and starts runs with the harness CLI (`enqueue`, `work`,
`export`), which also covers Ollama models and partial runs (some conditions or snapshots).

```bash
export ROOTLINE_DATA_DIR=~/RootLine/data
export ROOTLINE_REPO_DIR=~/RootLine
claude mcp add rootline -- uv run --directory /path/to/rootline-mcp rootline-mcp --stdio
```

## Deploy for free (Render) and add it to Claude

1. Put this folder in its own GitHub repository (for example `Flotrpy/rootline-mcp`).
2. Add `deploy/rootline-score.yml` to RootLine as `.github/workflows/rootline-score.yml`,
   and add the repository secrets `HF_TOKEN` (write access to `Flotrpy/rootline-work`)
   and `GROQ_API_KEY`.
3. Make a fine-grained GitHub token limited to `Flotrpy/RootLine` with
   **Actions: Read and write** and nothing else.
4. On Render, choose **New > Blueprint**, pick the repository and keep the free plan.
   Render reads `render.yaml`, generates `ROOTLINE_MCP_TOKEN` and asks for `HF_TOKEN`
   (read access is enough) and `ROOTLINE_GITHUB_TOKEN`. If Render ever asks for a card,
   stop: the free plan shouldn't need one.
5. In Claude, add a custom connector with the URL
   `https://<your-service>.onrender.com/<ROOTLINE_MCP_TOKEN>/mcp`.

The free plan sleeps after 15 idle minutes, so the first call after a quiet spell takes
about a minute while it wakes. Read tools work without any token at all (they use the
public dataset); starting runs is switched off unless `ROOTLINE_MCP_TOKEN` is set.

## Settings

| Variable | Meaning |
|---|---|
| `ROOTLINE_MCP_TOKEN` | Shared secret. The endpoint is `/<token>/mcp`, or `/mcp` with `Authorization: Bearer <token>`. |
| `HF_TOKEN` | Reads the private working dataset (and writes it for `merge_results(write=true)`). |
| `ROOTLINE_GITHUB_TOKEN` | Starts runs through the workflow. |
| `ROOTLINE_DATA_DIR`, `ROOTLINE_REPO_DIR` | Local data folder and local checkout instead of Hugging Face and GitHub. |
| `ROOTLINE_HF_DATASET`, `ROOTLINE_HF_WORK` | Dataset names (defaults `Flotrpy/rootline`, `Flotrpy/rootline-work`). |
| `ROOTLINE_GITHUB_REPO`, `ROOTLINE_GITHUB_WORKFLOW`, `ROOTLINE_GITHUB_REF` | Defaults `Flotrpy/RootLine`, `rootline-score.yml`, `main`. |
| `ROOTLINE_TZ` | Time zone for times shown (default `America/New_York`). |
| `ROOTLINE_MCP_MOCK` | `1` for made-up data and a pretend run backend. |

## Develop

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest -q          # 23 tests, no network needed
uv run ruff check src tests && uv run ruff format --check src tests
```

`tests/test_catalog_sync.py` checks the copied model list, faults and free-tier caps
against a RootLine checkout when one is present (`ROOTLINE_REPO_DIR`).

### Adding a feature

1. Put the logic in `src/rootline_mcp/tools.py` as a method on `Tools` (pure, testable).
2. Register it in `src/rootline_mcp/server.py` with a clear description and annotations
   (`READ` for anything that only reads).
3. Add tests in `tests/` against the mock data (`tests/conftest.py`), and extend
   `src/rootline_mcp/mock.py` if the feature needs data it doesn't have yet.
4. Add the tool name to the list in `tests/test_http.py`, a row to the table above and a
   line to `CHANGELOG.md`.

## Limits

- The connector's copy of the model list and free-tier caps must follow the harness;
  the sync test catches drift.
- On GitHub Actions a run always covers every snapshot and condition for one model,
  because that's what `score_daily.py` does. Partial runs need the local backend.
- Results are only as fresh as the last upload to the working dataset.
