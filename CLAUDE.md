Initiate all work with:
conda activate fastdl

## Project: dev-pipe

Unified golf swing analysis package. Merges the **eagle_swing library** (research/analysis modules) with **golf-pipeline-final** (production AWS pipeline) into a single, clean codebase under `dev-pipe/`.

## Scope Rules (CRITICAL — HARD CONSTRAINTS)

- **Read-only sources:** `eagle-swing/eagleswing/eagle_swing/` and `golf-pipeline-final/`. Never modify files in those directories.
- **Write target:** All new/modified code goes into `dev-pipe/` only.
- **Data path:** `/home/azaidi/Desktop/golf/data/full_videos/ymirza` (5 sessions: aug9, jun8, nov16, oct25, sep14). Never move, copy, or modify video/pkl data files. Symlink or configure paths.
- **No side effects:** Do not touch anything in the parent monorepo (`/home/azaidi/Desktop/fastai/testing/`) outside of `dev-pipe/`.
- **Git:** `dev-pipe/` will be its own git repo pushed to `github.com/azaidi06/dev-pipe`. Do not alter `.git` in eagle-swing or golf-pipeline-final.

## Shell Safety (MANDATORY — skip-permissions mode)

- **NEVER** run `rm -rf`, `rm -r`, or any recursive delete outside of `dev-pipe/`.
- **NEVER** run `git push --force`, `git reset --hard`, or `git clean -f` in any repo other than `dev-pipe/`.
- **NEVER** run `aws` CLI commands (s3, sqs, lambda, ec2, dynamodb, etc.). All AWS interaction is code-only, never shell.
- **NEVER** run any command that writes, moves, or deletes files outside of `/home/azaidi/Desktop/fastai/testing/dev-pipe/`.
- **NEVER** run `chmod`, `chown`, or permission changes on any path outside of `dev-pipe/`.
- **NEVER** run `git add -A` or `git add .` from any directory above `dev-pipe/`.
- **NEVER** run `pip install` without `--dry-run` first, except `pip install -e .` inside `dev-pipe/`.
- If unsure whether a command is safe, ask the user first.

## Architecture

```
dev-pipe/
  eagle_swing/            # Library package (installable via pip install -e .)
    __init__.py
    data/                 # Data loading, pkl I/O, metadata
      loader.py           # Merged from eagle_swing/data_class.py + pipeline pkl loaders
      schema.py           # Dataclasses: SwingMetaData, DetectionResult, ContactResult, etc.
    detection/            # Swing event detection
      backswing.py        # From golf-pipeline-final/swing_detection/detect.py
      contact.py          # Contact detection (from detect.py detect_contacts)
      events.py           # From eagle_swing/swing_events.py (address, takeaway, phases)
      config.py           # Merged detection configs
    analysis/             # Biomechanical analysis
      core.py             # From golf-pipeline-final/analyze/core.py (SwingData, metrics, SPM)
      upper_body.py       # From eagle_swing/upper_body.py
      lower_body.py       # From eagle_swing/lower_body.py
      kinematics.py       # From eagle_swing/kinematics.py
      normalization.py    # From eagle_swing/normalization.py
      temporal.py         # From eagle_swing/temporal.py
    hand_finder/          # Post-swing hand detection + finger count
      detect.py           # From golf-pipeline-final/hand_finder/detect.py
      predict.py          # From golf-pipeline-final/hand_finder/predict.py
      config.py           # From golf-pipeline-final/hand_finder/config.py
    viz/                  # All visualization
      plot.py             # From eagle_swing/plot.py + golf-pipeline-final/analyze/plots.py
      animate.py          # From eagle_swing/animate.py
      skeleton.py         # Skeleton rendering helpers
    pipeline/             # AWS pipeline orchestration (thin wrappers)
      label.py            # Calls into label_videos worker logic
      ingest.py           # Upload handler references
      lambda_handlers.py  # Detection + analysis lambda entry points
    utils.py              # Shared utilities
  pyproject.toml          # Package metadata, dependencies, entry points
  tests/                  # Merged test suite
    test_detection.py
    test_analysis.py
    test_hand_finder.py
    conftest.py           # Shared fixtures (synthetic pkl, ymirza paths)
  scripts/                # CLI entry points and batch runners
    run_detection.py      # Batch swing detection (from run_batch.py)
    run_analysis.py       # Analysis pipeline CLI (from analyze/pipeline.py)
    run_local_label.py    # Local labeling (from worker.py --local)
  docs/                   # Documentation site (Vite + React, deployable)
    src/
    public/
    package.json
    vite.config.js
  blog/                   # Quarto blog posts (ported from eagle-swing/posts/)
    _quarto.yml
    posts/
  artifacts/              # Generated outputs (gitignored, deployed to S3/hosting)
  CLAUDE.md               # This file
  README.md
```

## Integration Strategy

### Phase 1: Scaffold + Data Layer
1. Init git repo in `dev-pipe/`, create `pyproject.toml` with `eagle_swing` package
2. Port `data_class.py` (eagle_swing) + pkl loading (pipeline) into `eagle_swing/data/`
3. Unify pkl schema: eagle_swing uses `SwingKeypointData` wrapper; pipeline uses raw dict with `frame_N` keys. New `loader.py` handles both formats transparently
4. Config for ymirza data path: env var `EAGLE_SWING_DATA` defaulting to `/home/azaidi/Desktop/golf/data/full_videos/ymirza`
5. `conftest.py` with synthetic pkl fixtures + ymirza session discovery

### Phase 2: Detection Module
1. Port `swing_detection/detect.py` + `config.py` into `eagle_swing/detection/backswing.py`
2. Port `swing_events.py` (address, takeaway, phases) into `eagle_swing/detection/events.py`
3. Merge contact detection into `eagle_swing/detection/contact.py`
4. All detection functions accept either raw pkl path or `SwingKeypointData` objects
5. Tests: port `test_detect.py`, add ymirza integration tests

### Phase 3: Analysis Module
1. Port `analyze/core.py` → `eagle_swing/analysis/core.py`
2. Port `upper_body.py`, `lower_body.py`, `kinematics.py`, `normalization.py`, `temporal.py`
3. Port `analyze/plots.py` + eagle_swing `plot.py` → `eagle_swing/viz/plot.py`
4. Preserve `analyze/pipeline.py` CLI interface as `scripts/run_analysis.py`

### Phase 4: Hand Finder
1. Direct port of `hand_finder/` module into `eagle_swing/hand_finder/`
2. Keep predict.py EfficientNet-B0 model loading + inference
3. Maintain `FINGERS_DIR` env var for checkpoint path

### Phase 5: Pipeline Wrappers
1. Thin `eagle_swing/pipeline/` module that imports from label_videos, ingest, lambdas
2. These are orchestration entry points, not re-implementations
3. AWS resource configs stay as env vars / config.env files

### Phase 6: Docs + Blog + Artifacts
1. Port `golf-pipeline-final/docs/` React site into `dev-pipe/docs/`
2. Add new pages: library API reference, analysis examples, ymirza case study
3. Port relevant `eagle-swing/posts/` Quarto notebooks into `dev-pipe/blog/`
4. Deploy docs to GitHub Pages or AWS Amplify (mobile-accessible URL)
5. Deploy blog to Quarto Pub or GitHub Pages (separate subdomain/path)
6. Generated artifacts (plots, reports) uploaded to S3 with public URLs

## Deployment Targets

| Asset | Platform | URL Pattern |
|-------|----------|-------------|
| Docs site | GitHub Pages | `azaidi06.github.io/dev-pipe/` |
| Blog | Quarto Pub or GH Pages `/blog` | `azaidi06.github.io/dev-pipe/blog/` |
| Artifacts (plots, reports) | S3 + CloudFront | `artifacts.eagleswing.dev/` or S3 presigned URLs |
| Pipeline (label/detect) | AWS (EC2/Lambda) | Existing infra, no changes |

All URLs must be accessible from mobile.

## Key Design Decisions

1. **Library-first:** `eagle_swing` is a pip-installable package. Pipeline scripts import from it. This replaces the current split where eagle-swing is a Quarto blog and golf-pipeline-final is a standalone pipeline.
2. **Preserve pipeline CLAUDE.md patterns:** Frozen dataclass configs, pure numpy/scipy compute, no OpenCV in detection, matplotlib-only viz.
3. **PKL compatibility:** Both old eagle_swing format (list of dicts) and pipeline format (`frame_N` keyed dict with `__meta__`) must work.
4. **No duplicate code:** If both repos implement the same thing (e.g., wrist signal extraction, COCO keypoint maps), pick the better version and delete the other.
5. **Minimal deps:** Core library needs only numpy, scipy, pandas, matplotlib. Pipeline-specific deps (torch, mmcv, boto3) are optional extras in pyproject.toml.

## Naming Conventions

- Package: `eagle_swing` (underscore, importable)
- Repo: `dev-pipe` (dash, GitHub)
- Module files: snake_case
- Classes: PascalCase (SwingData, DetectionResult, Config)
- Functions: snake_case (detect_backswings, find_address_frame)

## Testing

- `pytest` from repo root
- AWS mocking via moto `@mock_aws`
- Synthetic pkl fixtures in `conftest.py`
- Integration tests use ymirza data (skip if not found: `@pytest.mark.skipif`)
- `pytest.ini` with `testpaths = ["tests"]`

## What NOT To Do

- Do not re-implement label_videos GPU worker. It stays in pipeline wrappers.
- Do not refactor AWS infra (S3 triggers, SQS, Lambda). That's deployment config.
- Do not add new ML models or training code.
- Do not modify the ymirza data files.
- Do not add Streamlit dashboards (prefer the React docs site for visualization).
- Do not over-abstract. If a function is used once, inline it.

## Subagent Strategy

Use subagents aggressively to parallelize work. The key insight: **research is cheap, writing is serial**. Front-load research across many agents, then write files sequentially from the results.

### Principles

1. **Research before writing.** Before porting any module, spawn an Explore agent to read the source file(s) thoroughly. Never write a ported module without first understanding every function, its callers, and its dependencies.
2. **Parallelize independent reads.** Phases 2, 3, and 4 have zero dependencies on each other (they all depend on Phase 1's data layer only). Their research can run simultaneously.
3. **One agent per source module.** Don't ask a single agent to read 5 files across 2 repos. Give each agent a focused scope: one module or one tightly-coupled pair.
4. **Write in the main context.** Subagents should return findings/analysis. The main thread writes files — this keeps dev-pipe's state consistent and avoids merge conflicts from parallel writes.
5. **Use background agents for long reads.** If an agent is reading a 500+ line file and you have other independent work, run it in background (`run_in_background: true`).

### Agent Playbook by Phase

**Phase 1 — Scaffold + Data Layer**
No subagents needed. The main thread creates `pyproject.toml`, `__init__.py` files, and directory structure. This is fast and sequential.

**Phase 2 — Detection Module (3 parallel Explore agents)**
```
Agent A: Read golf-pipeline-final/swing_detection/detect.py + config.py
         → Report: all functions, their signatures, internal helpers, scipy deps,
           Config dataclass fields, DetectionResult/ContactResult schemas

Agent B: Read eagle-swing/eagle_swing/swing_events.py
         → Report: all functions, what they compute, how they differ from
           pipeline's detect.py (overlap in wrist extraction? different algos?)

Agent C: Read golf-pipeline-final/swing_detection/tests/test_detect.py
         → Report: test structure, synthetic pkl format, assertions,
           what coverage exists
```
Then main thread writes `detection/backswing.py`, `detection/events.py`, `detection/config.py`, `tests/test_detection.py` using all three reports.

**Phase 3 — Analysis Module (3 parallel Explore agents)**
```
Agent D: Read golf-pipeline-final/analyze/core.py + plots.py
         → Report: SwingData class, all metrics, COCO_MAP, statistical methods,
           PlotConfig, render functions

Agent E: Read eagle-swing/eagle_swing/upper_body.py + lower_body.py + kinematics.py
         → Report: all body-segment functions, angle calculations, what overlaps
           with analyze/core.py metrics

Agent F: Read eagle-swing/eagle_swing/normalization.py + temporal.py
         → Report: normalization strategies, temporal alignment methods,
           what analyze/core.py's resampling duplicates
```
Main thread merges and deduplicates into `analysis/core.py`, `analysis/upper_body.py`, etc.

**Phase 4 — Hand Finder (1 Explore agent, background)**
```
Agent G: Read golf-pipeline-final/hand_finder/{config,detect,predict}.py
         → Report: full API, dependencies, model loading, config dataclasses
```
This is a clean port — one agent in background while main thread works on Phase 2/3 writes.

**Phase 5 — Pipeline Wrappers (1 Explore agent)**
```
Agent H: Read golf-pipeline-final/label_videos/worker.py (focus on process_local),
         ingest/handler.py, swing_detection/lambda_handler.py, analyze/lambda_handler.py
         → Report: entry points, what to wrap vs. what to copy
```

**Phase 6 — Docs + Blog (2 parallel agents)**
```
Agent I (Explore): Read golf-pipeline-final/docs/src/ — all pages, components,
         vite config, build setup
         → Report: what exists, what to port, what new pages to add

Agent J (Explore): Read eagle-swing/posts/ — scan all subdirectories,
         identify which notebooks are worth porting as blog posts
         → Report: ranked list of posts by relevance, their dependencies
```
Main thread ports docs site, then builds + deploys.

### Anti-Patterns

- **Don't spawn a write agent.** Two agents writing to `dev-pipe/` simultaneously will conflict. All file creation happens in the main thread.
- **Don't spawn agents for trivial reads.** If you need to check one function signature in a file you've already read, just use Read directly.
- **Don't chain agents.** Agent B should not depend on Agent A's output. If there's a dependency, wait for A to finish, then incorporate its findings into B's prompt.
- **Don't use worktree isolation.** `dev-pipe/` is new — there's nothing to isolate from. Worktrees add overhead for no benefit here.
- **Don't re-research.** Once an agent reports on a module, save its findings mentally. Don't spawn another agent to re-read the same file.

### Typical Session Flow

```
1. Main thread: Create scaffold (Phase 1)                    ~5 min
2. Spawn Agents A+B+C+D+E+F+G in parallel (research)        ~2 min wall clock
3. Collect results. Main thread writes Phase 2 files.         ~10 min
4. Main thread writes Phase 3 files (using D+E+F results).   ~10 min
5. Main thread writes Phase 4 files (using G result).         ~5 min
6. Spawn Agent H. Main thread writes Phase 5.                ~5 min
7. Spawn Agents I+J. Main thread writes Phase 6.             ~10 min
8. Build, test, deploy, commit.                               ~5 min
```

Peak parallelism: 7 concurrent Explore agents during step 2. This is the highest-leverage moment — it front-loads all the understanding needed for the rest of the session.

## Current Status

- [x] Phase 1: Scaffold + Data Layer
- [x] Phase 2: Detection Module
- [x] Phase 3: Analysis Module
- [x] Phase 4: Hand Finder
- [x] Phase 5: Pipeline Wrappers
- [x] Phase 6: Docs + Blog + Artifacts (scaffolded, deploy pending push to GitHub)
