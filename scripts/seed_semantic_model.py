"""Seed semantic model (field-level business mappings) into MySQL.

Usage:
    python -m scripts.seed_semantic_model
"""

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
    )


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS t_semantic_model (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                table_name VARCHAR(128) NOT NULL COMMENT '物理表名',
                column_name VARCHAR(128) NOT NULL COMMENT '物理字段名',
                business_name VARCHAR(256) COMMENT '业务名称',
                synonyms TEXT COMMENT '同义词，逗号分隔',
                business_description TEXT COMMENT '业务描述（枚举值、状态码、计算逻辑等）',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_table_col (table_name, column_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='语义模型：字段级业务映射'
        """)
    conn.commit()
    print("t_semantic_model table created")


def seed_data(conn):
    """Seed semantic model for financial tables."""
    records = [
        # t_account
        ("t_account", "code", "科目编码", "会计科目代码", "财务科目唯一编码，如 1001=库存现金"),
        ("t_account", "name", "科目名称", "会计科目, 账户名称", "财务科目名称"),
        ("t_account", "type", "科目类型", "账户类型", "资产/负债/所有者权益/成本/损益"),
        ("t_account", "balance_direction", "余额方向", "借贷方向", "借/贷"),
        ("t_account", "level", "科目层级", "", "1=一级科目, 2=二级科目"),

        # t_cost_center
        ("t_cost_center", "code", "成本中心编码", "部门编码", "成本中心唯一编码"),
        ("t_cost_center", "name", "成本中心名称", "部门名称", "如：研发部、市场部、财务部"),
        ("t_cost_center", "manager", "负责人", "部门经理, 主管", "成本中心负责人姓名"),
        ("t_cost_center", "annual_budget", "年度预算", "全年预算", "该成本中心的年度预算金额"),

        # t_journal_entry
        ("t_journal_entry", "entry_no", "凭证号", "记账凭证编号", "会计凭证唯一编号"),
        ("t_journal_entry", "entry_date", "凭证日期", "记账日期", "凭证录入日期"),
        ("t_journal_entry", "entry_type", "凭证类型", "", "收/付/转"),
        ("t_journal_entry", "period", "会计期间", "账期", "格式 YYYYMM，如 202601"),
        ("t_journal_entry", "debit_total", "借方合计", "", "凭证借方金额合计"),
        ("t_journal_entry", "credit_total", "贷方合计", "", "凭证贷方金额合计"),
        ("t_journal_entry", "status", "凭证状态", "", "draft=草稿, posted=已过账, voided=已作废"),
        ("t_journal_entry", "preparer", "制单人", "录入人", "凭证制单人姓名"),
        ("t_journal_entry", "reviewer", "审核人", "", "凭证审核人姓名"),

        # t_journal_item
        ("t_journal_item", "amount", "记账金额", "交易金额, 发生额, 借贷金额", "凭证行的借方或贷方金额"),
        ("t_journal_item", "direction", "借贷方向", "", "debit=借方, credit=贷方"),
        ("t_journal_item", "description", "摘要", "备注, 说明", "凭证行摘要说明"),
        ("t_journal_item", "cost_center_id", "成本中心", "部门", "关联的成本中心"),

        # t_fund_transfer
        ("t_fund_transfer", "transfer_no", "转账单号", "划款单号", "资金划转单号"),
        ("t_fund_transfer", "transfer_type", "转账类型", "划款类型", "internal=内部转账, external=外部转账"),
        ("t_fund_transfer", "amount", "转账金额", "划款金额", "资金划转金额"),
        ("t_fund_transfer", "status", "转账状态", "", "pending=待审批, approved=已批准, completed=已完成, rejected=已拒绝"),
        ("t_fund_transfer", "currency", "币种", "", "CNY/USD/EUR"),

        # t_budget
        ("t_budget", "year", "预算年度", "", "预算所属年份"),
        ("t_budget", "month", "预算月份", "", "预算所属月份，1-12"),
        ("t_budget", "budget_amount", "预算金额", "预算额度", "该月预算金额"),
        ("t_budget", "actual_amount", "实际金额", "实际发生额", "该月实际发生金额"),
        ("t_budget", "variance", "差异", "偏差", "实际 - 预算，正数=超支"),

        # t_invoice
        ("t_invoice", "invoice_no", "发票号码", "", "发票唯一号码"),
        ("t_invoice", "invoice_type", "发票类型", "", "增值税专用/普通/电子"),
        ("t_invoice", "amount", "发票金额", "不含税金额", "发票不含税金额"),
        ("t_invoice", "tax_amount", "税额", "", "发票税额"),
        ("t_invoice", "direction", "开票方向", "", "input=进项, output=销项"),
        ("t_invoice", "verify_status", "认证状态", "", "pending=待认证, verified=已认证, rejected=已退回"),

        # t_receivable_payable
        ("t_receivable_payable", "type", "类型", "", "receivable=应收, payable=应付"),
        ("t_receivable_payable", "counterparty", "往来单位", "对方单位, 客户/供应商", "交易对手方名称"),
        ("t_receivable_payable", "original_amount", "原始金额", "合同金额", "应收/应付原始金额"),
        ("t_receivable_payable", "settled_amount", "已结算金额", "已收/已付金额", "已结算的金额"),
        ("t_receivable_payable", "balance", "余额", "未结金额", "未结算余额 = 原始金额 - 已结算金额"),
        ("t_receivable_payable", "due_date", "到期日", "", "应收/应付到期日期"),
        ("t_receivable_payable", "status", "状态", "", "pending=待结算, partial=部分结算, settled=已结清, overdue=逾期"),

        # t_expense_claim
        ("t_expense_claim", "claim_no", "报销单号", "", "报销单编号"),
        ("t_expense_claim", "claimant", "报销人", "申请人", "报销申请人姓名"),
        ("t_expense_claim", "expense_type", "费用类型", "报销类型", "差旅/办公/招待/交通/培训"),
        ("t_expense_claim", "amount", "报销金额", "费用金额", "报销总金额"),
        ("t_expense_claim", "status", "报销状态", "", "pending=待审批, approved=已批准, paid=已付款, rejected=已拒绝"),
        ("t_expense_claim", "approver", "审批人", "", "报销审批人"),

        # t_fixed_asset
        ("t_fixed_asset", "asset_code", "资产编码", "资产编号", "固定资产唯一编码"),
        ("t_fixed_asset", "asset_name", "资产名称", "设备名称", "固定资产名称"),
        ("t_fixed_asset", "original_cost", "原值", "购置原值", "固定资产原始购置成本"),
        ("t_fixed_asset", "accumulated_depreciation", "累计折旧", "", "截至当前的累计折旧额"),
        ("t_fixed_asset", "net_value", "净值", "", "原值 - 累计折旧"),
        ("t_fixed_asset", "location", "存放地点", "使用部门", "资产存放位置"),
    ]

    with conn.cursor() as cur:
        for table_name, column_name, business_name, synonyms, description in records:
            cur.execute(
                """INSERT INTO t_semantic_model (table_name, column_name, business_name, synonyms, business_description)
                   VALUES (%s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                       business_name=VALUES(business_name),
                       synonyms=VALUES(synonyms),
                       business_description=VALUES(business_description)""",
                (table_name, column_name, business_name, synonyms, description),
            )
    conn.commit()
    print(f"Seeded {len(records)} semantic model entries")


def main():
    conn = get_conn()
    try:
        create_table(conn)
        seed_data(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
