# Data Permission and Audit V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first production-shaped data permission and audit loop to the NL2SQL path without hiding missing permissions from the user.

**Architecture:** Keep SQL generation based on physical schema, but add deterministic gates around it. Table permissions are checked after table selection and when SQL generation asks for missing tables; SQL authorization is checked before approval/execution; results are rendered with business field names before user display. Column masking, row-level predicates, and durable audit storage are explicitly deferred to the next iteration.

**Tech Stack:** Python 3.11, LangGraph, FastAPI, PyMySQL, existing `t_semantic_model`, local unit tests with pytest.

---

### Task 1: Security Policy Core

**Files:**
- Create: `agents/tool/security/policies.py`
- Create: `tests/test_security_policies.py`

- [x] Write failing tests for default allow, table denial, business display names, and audit payload shape.
- [x] Implement `SecurityContext`, `TableAuthorizationResult`, `authorize_tables()`, `display_name_for_table()`, and `build_audit_event()`.
- [x] Run policy tests through the focused verification suite.

### Task 2: Result Presentation Core

**Files:**
- Create: `agents/tool/security/presentation.py`
- Test: `tests/test_security_presentation.py`

- [x] Write failing tests that map duplicate physical columns through table-aware metadata and preserve already aliased Chinese columns.
- [x] Implement `build_column_display_map()` and `format_result_for_user()`.
- [x] Run presentation tests through the focused verification suite.

### Task 3: SQL Graph Gates

**Files:**
- Modify: `agents/flow/state.py`
- Modify: `agents/flow/sql_react.py`
- Test: `tests/test_sql_react.py`

- [x] Write failing tests proving unauthorized selected tables stop before `sql_retrieve`.
- [x] Write failing tests proving unauthorized `missing_tables` do not call schema retrieval.
- [x] Add `authorize_selected_tables` graph node between `select_tables` and `assess_feasibility`.
- [x] Add an authorization check before `_retrieve_missing_tables()`.
- [x] Add `authorize_sql` as a conservative V1 node after `safety_check`.
- [x] Ensure complex plan SQL steps also run `authorize_sql` before execution.
- [x] Run targeted SQL React tests through the focused verification suite.

### Task 4: API Security Context and Audit Skeleton

**Files:**
- Modify: `agents/api/routers/query.py`
- Create: `agents/tool/security/audit.py`
- Test: `tests/test_query_security_context.py`

- [x] Write failing tests that API state carries user/session security context.
- [x] Implement default context from `session_id` and optional headers.
- [x] Implement best-effort audit writer with a no-throw fallback.
- [x] Ensure permission denials are auditable even when the SQL graph stops early.

### Task 5: Verification

**Files:**
- All touched files.

- [x] Run focused tests:
  `pytest tests/test_security_policies.py tests/test_security_presentation.py tests/test_sql_react.py -q`
- [x] Run the broader stable suite if focused tests pass:
  `pytest -q`
- [x] Review `git diff` and update `docs/iterations.md` to separate V1 shipped scope from deferred SQL AST / row-level / masking work.
