# BofA OCC Coverage Demo Guide

This fork is prepared for a Devin executive demo around Bank of America OCC
readiness. Prefer narrowly scoped test-coverage work over broad refactors.

## Customer Scenario

Bank of America needs to demonstrate adequate coverage on compliance-critical
paths before an OCC regulatory exam. The demo focuses on four control areas:

- Transaction processing
- Authentication
- PII handling
- Audit logging

## Coverage Inventory Targets

Use these file/category pairs when asked to calculate or improve compliance
coverage:

| Category | Files |
| --- | --- |
| Transaction processing | `superset/sql/execution/executor.py`, `superset/sql/execution/celery_task.py`, `superset/utils/rls.py` |
| Authentication | `superset/security/manager.py`, `superset/security/api.py`, `superset/security/guest_token.py`, `superset/utils/rls.py`, `superset/security/analytics_db_safety.py` |
| PII handling | `superset/mcp_service/utils/sanitization.py`, `superset/mcp_service/middleware.py` |
| Audit logging | `superset/mcp_service/middleware.py`, `superset/security/manager.py`, `superset/sql/execution/executor.py` |

## Fast Coverage Command

For quick OCC coverage measurement, avoid writing artifacts into the repo:

```bash
export COVERAGE_FILE=/tmp/bofa-occ-coverage.$$.coverage
PYTHONDONTWRITEBYTECODE=1 pytest \
  tests/unit_tests/sql/execution/test_executor.py \
  tests/unit_tests/sql/execution/test_celery_task.py \
  tests/unit_tests/security/guest_rls_test.py \
  tests/unit_tests/security/api_test.py \
  tests/unit_tests/security/audit_log_test.py \
  tests/unit_tests/security/analytics_db_safety_test.py \
  tests/unit_tests/mcp_service/test_middleware.py \
  tests/unit_tests/mcp_service/test_middleware_logging.py \
  tests/unit_tests/mcp_service/utils/test_sanitization.py \
  --cov=superset \
  --cov-branch \
  --cov-report= \
  --no-cov-on-fail

python -m coverage report \
  --include="superset/sql/execution/executor.py,superset/sql/execution/celery_task.py,superset/utils/rls.py,superset/security/manager.py,superset/security/api.py,superset/security/guest_token.py,superset/security/analytics_db_safety.py,superset/mcp_service/utils/sanitization.py,superset/mcp_service/middleware.py"
rm -f "$COVERAGE_FILE"
```

When returning structured results, provide only one row per category/file pair
with `category`, `category_label`, `file_path`, `lines_of_code`, and
`coverage_percent`.

## Fix Guidance

When assigned a sub-80% coverage issue:

- Add targeted tests first.
- Make minimal code changes only when a real edge case blocks meaningful tests.
- Keep PRs reviewable for security and audit stakeholders.
- Include acceptance criteria evidence in the PR description.
