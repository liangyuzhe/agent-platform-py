# Full Online Evaluation Docs Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the full online preselect evaluation, then update project evaluation docs and the generated resume with measured metrics.

**Architecture:** Use the existing `agents.eval.cli run --include-online-pipeline` path to measure `recall_evidence -> query_enhance -> select_tables` on `data/eval/eval_dataset.jsonl`. Treat the resulting JSON report as the single source of truth for documentation and resume metrics.

**Tech Stack:** Python venv, pytest, jq, Markdown docs, DOCX resume.

---

### Task 1: Run Full Online Evaluation

**Files:**
- Read: `data/eval/eval_dataset.jsonl`
- Write: `data/eval/eval_report.json`

- [ ] **Step 1: Execute the online preselect evaluation**

Run:

```bash
.venv/bin/python -m agents.eval.cli run \
  --dataset data/eval/eval_dataset.jsonl \
  --output data/eval/eval_report.json \
  --include-online-pipeline
```

Expected: command completes and prints a report including `preselect_pipeline`.

- [ ] **Step 2: Extract metrics**

Run:

```bash
jq '.strategies[] | select(.strategy=="preselect_pipeline") | {num_queries, metrics, latency}' data/eval/eval_report.json
```

Expected: JSON with `recall@5`, `mrr`, `accuracy@5`, and latency values.

### Task 2: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/evaluation_design.md`
- Modify: `docs/evaluation_user_guide.md`
- Modify: `docs/iterations.md`

- [ ] **Step 1: Locate stale metric references**

Run:

```bash
rg -n "82\\.48|67\\.59|Recall@5|MRR|preselect_pipeline|management_preselect" README.md docs/evaluation_design.md docs/evaluation_user_guide.md docs/iterations.md
```

Expected: list of places to update.

- [ ] **Step 2: Patch docs with fresh report metrics**

Update metrics to distinguish:
- full online evaluation on `eval_dataset.jsonl`
- management-only online evaluation on `management_eval_dataset.jsonl`
- offline schema lexical baseline

- [ ] **Step 3: Verify docs no longer contain stale claims**

Run:

```bash
rg -n "67\\.59|Recall@5 = 100%.*全量|整体.*100%" README.md docs/evaluation_design.md docs/evaluation_user_guide.md docs/iterations.md
```

Expected: no misleading full-evaluation claims.

### Task 3: Update Resume

**Files:**
- Modify: `outputs/resume/梁宇哲_后端工程师_NL2SQL项目版.docx`

- [ ] **Step 1: Inspect DOCX text**

Run a Python script using `python-docx` if available to extract current project bullets.

- [ ] **Step 2: Update metric bullets**

Replace stale metrics with the new full online evaluation result and keep management-table recall as a scoped专项 result.

- [ ] **Step 3: Verify DOCX contains fresh metrics**

Extract text again and check for the expected values.

### Task 4: Final Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_eval_pipeline_strategies.py tests/test_sql_react.py::TestSelectTables tests/test_retriever_relationships.py tests/test_seed_semantic_model.py
```

Expected: all tests pass.

- [ ] **Step 2: Check formatting**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.
