"""Seed agent knowledge (SQL Q&A few-shot pairs) into MySQL + Milvus.

SQL column names match seed_financial.py table definitions exactly.

Usage:
    python -m scripts.seed_agent_knowledge
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
            CREATE TABLE IF NOT EXISTS t_agent_knowledge (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                question TEXT NOT NULL COMMENT '用户问题',
                sql_text TEXT NOT NULL COMMENT '参考 SQL',
                description TEXT COMMENT '说明',
                category VARCHAR(64) COMMENT '分类: query/report/analysis',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uk_question (question(128))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='智能体知识库：SQL Q&A few-shot 对'
        """)
    conn.commit()
    print("t_agent_knowledge table created")


def seed_data(conn):
    records = [
        # 基础查询
        ("查询所有科目余额",
         "SELECT a.account_code, a.account_name, a.account_type, SUM(ji.debit_amount) - SUM(ji.credit_amount) AS balance FROM t_journal_item ji JOIN t_account a ON ji.account_code = a.account_code GROUP BY a.account_code, a.account_name, a.account_type",
         "按科目汇总借方减贷方得到余额", "query"),

        ("本月各部门费用汇总",
         "SELECT cc.center_name AS department, SUM(ec.total_amount) AS total_expense FROM t_expense_claim ec JOIN t_cost_center cc ON ec.cost_center_id = cc.id WHERE ec.status IN ('已审批', '已付款') GROUP BY cc.id, cc.center_name ORDER BY total_expense DESC",
         "已批准的报销按部门汇总", "query"),

        ("查询所有未过账的凭证",
         "SELECT je.entry_no, je.entry_date, je.entry_type, je.total_debit, je.total_credit, je.prepared_by FROM t_journal_entry je WHERE je.status = '草稿' ORDER BY je.entry_date",
         "筛选草稿状态的凭证", "query"),

        ("查询逾期应收款项",
         "SELECT rp.counterparty, rp.original_amount, rp.settled_amount, rp.original_amount - rp.settled_amount AS balance, rp.due_date FROM t_receivable_payable rp WHERE rp.rp_type = '应收' AND rp.status = '逾期' ORDER BY rp.due_date",
         "筛选逾期状态的应收账款", "query"),

        ("查询本月预算执行情况",
         "SELECT b.budget_month, cc.center_name AS department, b.budget_amount, b.actual_amount, b.actual_amount - b.budget_amount AS variance, ROUND(b.actual_amount / b.budget_amount * 100, 2) AS execution_rate FROM t_budget b JOIN t_cost_center cc ON b.cost_center_id = cc.id WHERE b.budget_year = 2025 ORDER BY b.budget_month, cc.center_name",
         "预算与实际对比，计算执行率", "analysis"),

        ("查询固定资产净值",
         "SELECT fa.asset_code, fa.asset_name, fa.acquisition_cost, fa.accumulated_depreciation, fa.acquisition_cost - fa.accumulated_depreciation AS net_value, ROUND((fa.acquisition_cost - fa.accumulated_depreciation) / fa.acquisition_cost * 100, 2) AS net_value_rate FROM t_fixed_asset fa ORDER BY net_value DESC",
         "固定资产净值及净值率", "query"),

        ("查询待认证发票",
         "SELECT i.invoice_no, i.invoice_type, i.amount_without_tax, i.tax_amount, i.total_amount FROM t_invoice i WHERE i.verification_status = '未认证' AND i.direction = '进项' ORDER BY i.amount_without_tax DESC",
         "进项发票中待认证的", "query"),

        ("查询本月资金划转记录",
         "SELECT ft.transfer_no, ft.transfer_type, ft.amount, ft.status, ft.transfer_date FROM t_fund_transfer ft WHERE ft.transfer_date >= '2025-06-01' AND ft.transfer_date < '2025-07-01' ORDER BY ft.transfer_date",
         "当月资金划转流水", "query"),

        ("查询各部门年度预算总额",
         "SELECT cc.center_name AS department, SUM(b.budget_amount) AS annual_budget, SUM(b.actual_amount) AS annual_actual, ROUND(SUM(b.actual_amount) / SUM(b.budget_amount) * 100, 2) AS execution_rate FROM t_budget b JOIN t_cost_center cc ON b.cost_center_id = cc.id WHERE b.budget_year = 2025 GROUP BY cc.id, cc.center_name ORDER BY execution_rate DESC",
         "年度预算执行率排名", "analysis"),

        ("查询报销金额最大的前10笔",
         "SELECT ec.claim_no, ec.claimant, ec.expense_type, ec.total_amount, ec.claim_date FROM t_expense_claim ec WHERE ec.status IN ('已审批', '已付款') ORDER BY ec.total_amount DESC LIMIT 10",
         "Top 10 报销记录", "report"),

        ("查询凭证过账率",
         "SELECT COUNT(CASE WHEN status = '已过账' THEN 1 END) AS posted_count, COUNT(*) AS total_count, ROUND(COUNT(CASE WHEN status = '已过账' THEN 1 END) / COUNT(*) * 100, 2) AS post_rate FROM t_journal_entry",
         "凭证过账率统计", "analysis"),

        ("查询各科目类型余额汇总",
         "SELECT a.account_type, SUM(ji.debit_amount) - SUM(ji.credit_amount) AS balance FROM t_journal_item ji JOIN t_account a ON ji.account_code = a.account_code GROUP BY a.account_type ORDER BY balance DESC",
         "按科目类型汇总余额", "analysis"),

        ("查询本年各月费用趋势",
         "SELECT DATE_FORMAT(claim_date, '%Y-%m') AS month, expense_type, SUM(total_amount) AS total FROM t_expense_claim WHERE status IN ('已审批', '已付款') GROUP BY month, expense_type ORDER BY month, total DESC",
         "按月按类型费用趋势", "report"),

        ("查询供应商应付账款",
         "SELECT counterparty, SUM(original_amount) AS total_amount, SUM(settled_amount) AS settled, SUM(original_amount - settled_amount) AS outstanding FROM t_receivable_payable WHERE rp_type = '应付' GROUP BY counterparty ORDER BY outstanding DESC",
         "供应商应付余额排名", "query"),

        ("查询借方金额最大的凭证分录",
         "SELECT ji.account_code, ji.summary, ji.debit_amount, ji.credit_amount, je.entry_no, je.entry_date FROM t_journal_item ji JOIN t_journal_entry je ON ji.entry_id = je.id ORDER BY ji.debit_amount DESC LIMIT 10",
         "Top 10 借方金额分录", "report"),
    ]

    with conn.cursor() as cur:
        for question, sql_text, description, category in records:
            cur.execute(
                """INSERT INTO t_agent_knowledge (question, sql_text, description, category)
                   VALUES (%s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                       sql_text=VALUES(sql_text),
                       description=VALUES(description),
                       category=VALUES(category)""",
                (question, sql_text, description, category),
            )
    conn.commit()
    print(f"Seeded {len(records)} agent knowledge entries")


async def index_to_milvus():
    """Vectorize agent knowledge Q&A pairs and store in Milvus."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT id, question, sql_text, description, category FROM t_agent_knowledge")
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No agent knowledge to index")
        return

    from langchain_core.documents import Document
    from agents.rag.retriever import _get_embeddings
    from pymilvus import MilvusClient

    docs = []
    for row in rows:
        content = f"问题: {row['question']}\nSQL: {row['sql_text']}"
        if row.get("description"):
            content += f"\n说明: {row['description']}"
        docs.append(Document(
            page_content=content,
            metadata={
                "source": "agent_knowledge",
                "question": row["question"],
                "category": row.get("category", ""),
                "doc_id": f"ak_{row['id']}",
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
            "source": "agent_knowledge",
            "table_name": "",
            "doc_id": doc_id,
        })

    client.insert(collection_name=settings.milvus.collection_name, data=records)
    client.close()
    print(f"Indexed {len(docs)} agent knowledge entries into Milvus")


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
