"""Unified seed script: create all tables and seed all data.

Runs in order:
  1. Business tables + data (seed_financial)
  2. Semantic model (seed_semantic_model)
  3. Business knowledge + Milvus index (seed_business_knowledge)
  4. Agent knowledge + Milvus index (seed_agent_knowledge)

Usage:
    python -m scripts.seed_all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from agents.model.chat_model import init_chat_models
    init_chat_models()

    # Step 1: Business tables + data
    print("=" * 60)
    print("Step 1/4: Creating business tables and seeding data")
    print("=" * 60)
    from scripts.seed_financial import main as seed_financial_main
    seed_financial_main()

    # Step 2: Semantic model
    print("=" * 60)
    print("Step 2/4: Seeding semantic model")
    print("=" * 60)
    from scripts.seed_semantic_model import main as seed_semantic_main
    seed_semantic_main()

    # Step 3: Business knowledge
    print("=" * 60)
    print("Step 3/4: Seeding business knowledge + Milvus index")
    print("=" * 60)
    from scripts.seed_business_knowledge import main as seed_bk_main
    seed_bk_main()

    # Step 4: Agent knowledge
    print("=" * 60)
    print("Step 4/4: Seeding agent knowledge + Milvus index")
    print("=" * 60)
    from scripts.seed_agent_knowledge import main as seed_ak_main
    seed_ak_main()

    print("\n" + "=" * 60)
    print("All seeding complete!")
    print("=" * 60)
    print("\nData summary:")
    _print_summary()


def _print_summary():
    """Print a summary of seeded data."""
    import pymysql
    from agents.config.settings import settings

    conn = pymysql.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.username,
        password=settings.mysql.password,
        database=settings.mysql.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    tables = [
        ("t_account", "会计科目"),
        ("t_cost_center", "成本中心"),
        ("t_journal_entry", "记账凭证"),
        ("t_journal_item", "凭证分录"),
        ("t_fund_transfer", "资金划转"),
        ("t_budget", "预算管理"),
        ("t_invoice", "发票"),
        ("t_receivable_payable", "应收应付"),
        ("t_expense_claim", "费用报销"),
        ("t_fixed_asset", "固定资产"),
        ("t_semantic_model", "语义模型"),
        ("t_business_knowledge", "业务知识"),
        ("t_agent_knowledge", "智能体知识库"),
    ]
    try:
        with conn.cursor() as cur:
            for table_name, desc in tables:
                try:
                    cur.execute(f"SELECT COUNT(*) AS cnt FROM {table_name}")
                    row = cur.fetchone()
                    print(f"  {desc:12s} ({table_name:25s}): {row['cnt']:>5d} 条")
                except Exception:
                    print(f"  {desc:12s} ({table_name:25s}): 表不存在")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
