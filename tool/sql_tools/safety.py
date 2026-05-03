"""SQL safety checker that inspects queries for destructive or risky patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class SafetyReport:
    """Result of a safety check on a SQL statement."""

    is_safe: bool = True
    risks: List[str] = field(default_factory=list)
    estimated_rows: int | None = None
    required_permissions: List[str] = field(default_factory=list)


class SQLSafetyChecker:
    """Static analysis of SQL statements for safety concerns."""

    # Each entry: (compiled regex, human-readable risk description, permission hint)
    DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
        (
            re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
            "DROP TABLE will permanently delete a table and all its data.",
            "DDL",
        ),
        (
            re.compile(r"\bDROP\s+DATABASE\b", re.IGNORECASE),
            "DROP DATABASE will permanently delete an entire database.",
            "DDL",
        ),
        (
            re.compile(r"\bTRUNCATE\b", re.IGNORECASE),
            "TRUNCATE removes all rows from a table without logging individual row deletions.",
            "DML",
        ),
        (
            re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE),
            "DELETE statement detected; ensure a WHERE clause limits affected rows.",
            "DML",
        ),
        (
            re.compile(r"\bUPDATE\b.+\bSET\b", re.IGNORECASE),
            "UPDATE statement detected; ensure a WHERE clause limits affected rows.",
            "DML",
        ),
        (
            re.compile(r"\bALTER\s+TABLE\b", re.IGNORECASE),
            "ALTER TABLE will modify the schema of an existing table.",
            "DDL",
        ),
        (
            re.compile(r"\bINSERT\s+INTO\b", re.IGNORECASE),
            "INSERT statement detected; verify the target table and values.",
            "DML",
        ),
        (
            re.compile(r"\bCREATE\s+TABLE\b", re.IGNORECASE),
            "CREATE TABLE will add a new table to the database.",
            "DDL",
        ),
        (
            re.compile(r"\bGRANT\b", re.IGNORECASE),
            "GRANT statement modifies database permissions.",
            "DCL",
        ),
        (
            re.compile(r"\bREVOKE\b", re.IGNORECASE),
            "REVOKE statement removes database permissions.",
            "DCL",
        ),
        (
            re.compile(r"\bLOAD_FILE\b|\bINTO\s+OUTFILE\b", re.IGNORECASE),
            "File I/O functions can read/write server filesystem files.",
            "FILE",
        ),
    ]

    # Patterns that indicate a DELETE/UPDATE **without** a WHERE clause.
    _DELETE_NO_WHERE = re.compile(
        r"\bDELETE\s+FROM\b(?!\s+.+\bWHERE\b)", re.IGNORECASE
    )
    _UPDATE_NO_WHERE = re.compile(
        r"\bUPDATE\b.+\bSET\b(?!\s+.+\bWHERE\b)", re.IGNORECASE
    )

    def check(self, sql: str) -> SafetyReport:
        """Analyse *sql* and return a :class:`SafetyReport`.

        The method performs regex-based static analysis only -- it does **not**
        execute or explain the query.
        """
        report = SafetyReport()
        seen_perms: set[str] = set()

        for pattern, risk_msg, perm in self.DANGEROUS_PATTERNS:
            if pattern.search(sql):
                report.is_safe = False
                report.risks.append(risk_msg)
                seen_perms.add(perm)

        # Extra heuristic: DELETE / UPDATE without WHERE is especially risky.
        if self._DELETE_NO_WHERE.search(sql):
            if "DELETE without WHERE clause" not in report.risks:
                report.risks.append(
                    "DELETE statement has no WHERE clause -- this will affect all rows."
                )
                report.is_safe = False

        if self._UPDATE_NO_WHERE.search(sql):
            if "UPDATE without WHERE clause" not in report.risks:
                report.risks.append(
                    "UPDATE statement has no WHERE clause -- this will affect all rows."
                )
                report.is_safe = False

        # Heuristic row estimation for simple SELECT * FROM <table> LIMIT n.
        limit_match = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
        if limit_match:
            report.estimated_rows = int(limit_match.group(1))

        report.required_permissions = sorted(seen_perms)
        return report
