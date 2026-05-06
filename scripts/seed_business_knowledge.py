"""Seed business knowledge (terms, formulas, synonyms) into MySQL + Milvus.

Usage:
    python -m scripts.seed_business_knowledge
"""

import asyncio
import sys
from pathlib import Path

import pymysql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.config.settings import settings


def get_conn():
    return pymysql.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.username,
        password=settings.mysql.password,
        database=settings.mysql.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS t_business_knowledge (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                term VARCHAR(128) NOT NULL COMMENT '业务术语',
                formula TEXT NOT NULL COMMENT '公式/定义',
                synonyms TEXT COMMENT '同义词，逗号分隔',
                related_tables TEXT COMMENT '关联表名，逗号分隔',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uk_term (term)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='业务知识：术语、公式、同义词'
        """)
    conn.commit()
    print("t_business_knowledge table created")


def seed_data(conn):
    records = [
        ("毛利率", "(SUM(credit_amount) - SUM(debit_amount)) / SUM(credit_amount) * 100 WHERE account_type='损益'", "gross margin, 毛利比率", "t_journal_item,t_account"),
        ("净利率", "(收入 - 成本 - 费用) / 收入 * 100", "net margin, 净利润比率", "t_journal_item,t_account,t_expense_claim"),
        ("预算执行率", "actual_amount / budget_amount * 100", "预算完成率, 执行进度, 执行比例", "t_budget"),
        ("预算差异", "actual_amount - budget_amount", "预算偏差, 超支金额", "t_budget"),
        ("资产负债率", "负债总额 / 资产总额 * 100", "负债率, 杠杆率", "t_account"),
        ("应收账款周转率", "收入 / 平均应收账款", "应收周转, 回款效率", "t_receivable_payable,t_journal_item"),
        ("应收逾期率", "逾期应收金额 / 应收总额 * 100", "逾期比例, 坏账率", "t_receivable_payable"),
        ("费用总额", "SUM(total_amount) FROM t_expense_claim WHERE status IN ('已审批','已付款')", "总费用, 费用合计, 报销总额", "t_expense_claim"),
        ("部门费用", "SUM(total_amount) GROUP BY cost_center_id", "各部门费用, 部门开销", "t_expense_claim,t_cost_center"),
        ("凭证过账率", "COUNT(CASE WHEN status='已过账' THEN 1 END) / COUNT(*) * 100", "过账比例", "t_journal_entry"),
        ("发票认证率", "COUNT(CASE WHEN verification_status='已认证' THEN 1 END) / COUNT(*) * 100 WHERE direction='进项'", "认证比例, 抵扣率", "t_invoice"),
        ("资金划转频率", "COUNT(*) / 天数", "转账频次, 划款频率", "t_fund_transfer"),
        ("固定资产净值率", "(acquisition_cost - accumulated_depreciation) / acquisition_cost * 100", "资产新旧程度", "t_fixed_asset"),
        ("制证人工作量", "COUNT(*) GROUP BY prepared_by", "凭证制作量, 制单统计", "t_journal_entry"),
        ("科目余额", "SUM(debit_amount) - SUM(credit_amount) GROUP BY account_code", "账户余额, 科目结余", "t_journal_item,t_account"),
    ]

    with conn.cursor() as cur:
        for term, formula, synonyms, related_tables in records:
            cur.execute(
                """INSERT INTO t_business_knowledge (term, formula, synonyms, related_tables)
                   VALUES (%s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                       formula=VALUES(formula),
                       synonyms=VALUES(synonyms),
                       related_tables=VALUES(related_tables)""",
                (term, formula, synonyms, related_tables),
            )
    conn.commit()
    print(f"Seeded {len(records)} business knowledge entries")


async def index_to_milvus():
    """Vectorize business knowledge and store in Milvus."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT term, formula, synonyms, related_tables FROM t_business_knowledge")
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No business knowledge to index")
        return

    from langchain_core.documents import Document
    from agents.rag.retriever import _get_embeddings
    from pymilvus import MilvusClient

    docs = []
    for row in rows:
        content = f"术语: {row['term']}\n公式: {row['formula']}"
        if row.get("synonyms"):
            content += f"\n同义词: {row['synonyms']}"
        if row.get("related_tables"):
            content += f"\n关联表: {row['related_tables']}"
        docs.append(Document(
            page_content=content,
            metadata={
                "source": "business_knowledge",
                "term": row["term"],
                "doc_id": f"bk_{row['term']}",
            },
        ))

    embeddings = _get_embeddings()
    milvus_uri = f"http://{settings.milvus.addr}"
    client = MilvusClient(uri=milvus_uri)

    doc_ids = [d.metadata["doc_id"] for d in docs]
    try:
        client.delete(collection_name=settings.milvus.collection_name, ids=doc_ids)
    except Exception:
        pass

    texts = [d.page_content for d in docs]
    vectors = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        vectors.extend(embeddings.embed_documents(texts[i : i + batch_size]))

    records = []
    for doc, doc_id, vector in zip(docs, doc_ids, vectors):
        records.append({
            "pk": doc_id,
            "text": doc.page_content,
            "vector": vector,
            "source": "business_knowledge",
            "table_name": "",
            "doc_id": doc_id,
        })

    client.insert(collection_name=settings.milvus.collection_name, data=records)
    client.close()
    print(f"Indexed {len(docs)} business knowledge entries into Milvus")


def main():
    conn = get_conn()
    try:
        create_table(conn)
        seed_data(conn)
    finally:
        conn.close()

    asyncio.run(index_to_milvus())


if __name__ == "__main__":
    main()
