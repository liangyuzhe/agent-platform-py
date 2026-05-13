"""Tests for schema relationship loading."""


def test_get_table_relationships_merges_physical_and_semantic_fks(monkeypatch):
    import pymysql
    from agents.rag.retriever import get_table_relationships

    queries = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params):
            queries.append((sql, params))

        def fetchall(self):
            sql = queries[-1][0]
            if "information_schema.key_column_usage" in sql:
                return [
                    {
                        "from_table": "t_user_role",
                        "from_column": "user_id",
                        "to_table": "t_user",
                        "to_column": "id",
                    }
                ]
            if "FROM t_semantic_model" in sql:
                return [
                    {
                        "from_table": "t_user_role",
                        "from_column": "user_id",
                        "to_table": "t_user",
                        "to_column": "id",
                    },
                    {
                        "from_table": "t_user_department",
                        "from_column": "department_id",
                        "to_table": "t_department",
                        "to_column": "id",
                    },
                ]
            return []

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def close(self):
            pass

    monkeypatch.setattr(pymysql, "connect", lambda **_kwargs: FakeConn())

    relationships = get_table_relationships(["t_user", "t_department"])

    assert relationships == [
        {
            "from_table": "t_user_role",
            "from_column": "user_id",
            "to_table": "t_user",
            "to_column": "id",
        },
        {
            "from_table": "t_user_department",
            "from_column": "department_id",
            "to_table": "t_department",
            "to_column": "id",
        },
    ]


def test_get_table_relationships_keeps_physical_fks_when_semantic_query_fails(monkeypatch):
    import pymysql
    from agents.rag.retriever import get_table_relationships

    queries = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params):
            queries.append((sql, params))
            if "FROM t_semantic_model" in sql:
                raise RuntimeError("missing semantic model table")

        def fetchall(self):
            return [
                {
                    "from_table": "t_journal_item",
                    "from_column": "entry_id",
                    "to_table": "t_journal_entry",
                    "to_column": "id",
                }
            ]

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def close(self):
            pass

    monkeypatch.setattr(pymysql, "connect", lambda **_kwargs: FakeConn())

    relationships = get_table_relationships(["t_journal_item"])

    assert relationships == [
        {
            "from_table": "t_journal_item",
            "from_column": "entry_id",
            "to_table": "t_journal_entry",
            "to_column": "id",
        }
    ]


def test_get_table_relationships_accepts_tuple_fetchall_results(monkeypatch):
    import pymysql
    from agents.rag.retriever import get_table_relationships

    queries = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params):
            queries.append((sql, params))

        def fetchall(self):
            sql = queries[-1][0]
            if "information_schema.key_column_usage" in sql:
                return ()
            if "FROM t_semantic_model" in sql:
                return (
                    {
                        "from_table": "t_user_role",
                        "from_column": "role_id",
                        "to_table": "t_role",
                        "to_column": "id",
                    },
                )
            return ()

    class FakeConn:
        def cursor(self):
            return FakeCursor()

        def close(self):
            pass

    monkeypatch.setattr(pymysql, "connect", lambda **_kwargs: FakeConn())

    relationships = get_table_relationships(["t_role"])

    assert relationships == [
        {
            "from_table": "t_user_role",
            "from_column": "role_id",
            "to_table": "t_role",
            "to_column": "id",
        }
    ]
