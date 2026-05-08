"""Seed financial business tables into MySQL.

Generates data relative to current date so queries like "last month" always work.

Usage:
    python -m scripts.seed_financial
"""

import random
import sys
from datetime import date, timedelta
from decimal import Decimal
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


def run_sql(cursor, sql, desc=""):
    try:
        cursor.execute(sql)
        if desc:
            print(f"  [OK] {desc}")
    except Exception as e:
        print(f"  [ERR] {desc}: {e}")


def _rand_date(year, month):
    """Generate a random date within the given month."""
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-{random.randint(1, last_day):02d}"


def _recent_months(n=6):
    """Return list of (year, month) for the last n months including current."""
    today = date.today()
    months = []
    for i in range(n - 1, -1, -1):
        y = today.year
        m = today.month - i
        while m <= 0:
            m += 12
            y -= 1
        months.append((y, m))
    return months


def _insert_journal_entry(cursor, entry_no, entry_date, entry_type, period, items, memo):
    """Insert a balanced journal entry and its line items."""
    total_debit = sum(item["debit"] for item in items)
    total_credit = sum(item["credit"] for item in items)
    cursor.execute(
        """
        INSERT INTO t_journal_entry (
            entry_no, entry_date, entry_type, period,
            total_debit, total_credit, attachment_count, status,
            prepared_by, reviewed_by, posted_at, memo
        )
        VALUES (
            %s, %s, %s, %s,
            %s, %s, 1, '已过账',
            '系统初始化', '系统审核', %s, %s
        )
        """,
        (
            entry_no,
            entry_date,
            entry_type,
            period,
            total_debit,
            total_credit,
            f"{entry_date} 10:00:00",
            memo,
        ),
    )
    entry_id = cursor.lastrowid

    for line_no, item in enumerate(items, 1):
        cursor.execute(
            """
            INSERT INTO t_journal_item (
                entry_id, line_no, account_code, summary,
                debit_amount, credit_amount, cost_center_id, project_code
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, NULL)
            """,
            (
                entry_id,
                line_no,
                item["account_code"],
                item["summary"],
                item["debit"],
                item["credit"],
                item.get("cost_center_id"),
            ),
        )


def seed_prior_year_loss_scenario(conn, cursor):
    """Seed deterministic prior-year P&L vouchers for NL2SQL loss tests."""
    loss_year = date.today().year - 1
    entry_prefix = f"LOSS-{loss_year}-%"

    cursor.execute(
        """
        DELETE ji
        FROM t_journal_item ji
        INNER JOIN t_journal_entry je ON ji.entry_id = je.id
        WHERE je.entry_no LIKE %s
        """,
        (entry_prefix,),
    )
    cursor.execute("DELETE FROM t_journal_entry WHERE entry_no LIKE %s", (entry_prefix,))

    monthly_scenarios = [
        (1, Decimal("80000.00"), Decimal("85000.00"), Decimal("38000.00")),
        (2, Decimal("92000.00"), Decimal("98000.00"), Decimal("41000.00")),
        (3, Decimal("105000.00"), Decimal("112000.00"), Decimal("43000.00")),
        (4, Decimal("118000.00"), Decimal("124000.00"), Decimal("45000.00")),
        (5, Decimal("126000.00"), Decimal("135000.00"), Decimal("47000.00")),
        (6, Decimal("132000.00"), Decimal("141000.00"), Decimal("50000.00")),
        (7, Decimal("128000.00"), Decimal("139000.00"), Decimal("48000.00")),
        (8, Decimal("121000.00"), Decimal("131000.00"), Decimal("46000.00")),
        (9, Decimal("116000.00"), Decimal("126000.00"), Decimal("44000.00")),
        (10, Decimal("110000.00"), Decimal("119000.00"), Decimal("42000.00")),
        (11, Decimal("98000.00"), Decimal("106000.00"), Decimal("39000.00")),
        (12, Decimal("90000.00"), Decimal("97000.00"), Decimal("37000.00")),
    ]

    totals = {"income": Decimal("0.00"), "cost": Decimal("0.00"), "expense": Decimal("0.00")}
    for month, income, cost, expense in monthly_scenarios:
        period = f"{loss_year}-{month:02d}"
        entry_date = f"{period}-25"
        totals["income"] += income
        totals["cost"] += cost
        totals["expense"] += expense

        _insert_journal_entry(
            cursor,
            f"LOSS-{loss_year}{month:02d}-INC",
            entry_date,
            "收款",
            period,
            [
                {"account_code": "1002", "summary": "确认主营业务收入", "debit": income, "credit": Decimal("0.00"), "cost_center_id": 4},
                {"account_code": "6001", "summary": "确认主营业务收入", "debit": Decimal("0.00"), "credit": income, "cost_center_id": 4},
            ],
            "上一年度亏损场景-收入",
        )
        _insert_journal_entry(
            cursor,
            f"LOSS-{loss_year}{month:02d}-COST",
            entry_date,
            "转账",
            period,
            [
                {"account_code": "6401", "summary": "结转主营业务成本", "debit": cost, "credit": Decimal("0.00"), "cost_center_id": 6},
                {"account_code": "1002", "summary": "结转主营业务成本", "debit": Decimal("0.00"), "credit": cost, "cost_center_id": 6},
            ],
            "上一年度亏损场景-成本",
        )
        _insert_journal_entry(
            cursor,
            f"LOSS-{loss_year}{month:02d}-EXP",
            entry_date,
            "付款",
            period,
            [
                {"account_code": "5401", "summary": "确认期间费用", "debit": expense, "credit": Decimal("0.00"), "cost_center_id": 2},
                {"account_code": "1002", "summary": "确认期间费用", "debit": Decimal("0.00"), "credit": expense, "cost_center_id": 2},
            ],
            "上一年度亏损场景-费用",
        )

    conn.commit()
    net_profit = totals["income"] - totals["cost"] - totals["expense"]
    loss_amount = abs(net_profit) if net_profit < 0 else Decimal("0.00")
    print(
        "  [OK] prior-year loss scenario "
        f"({loss_year}): income={totals['income']:.2f}, "
        f"cost={totals['cost']:.2f}, expense={totals['expense']:.2f}, "
        f"net_profit={net_profit:.2f}, loss_amount={loss_amount:.2f}"
    )


def create_tables(cursor):
    print("=== Creating tables ===")

    run_sql(cursor, """
        CREATE TABLE IF NOT EXISTS t_account (
            id INT PRIMARY KEY AUTO_INCREMENT,
            account_code VARCHAR(20) NOT NULL UNIQUE COMMENT '科目编码',
            account_name VARCHAR(100) NOT NULL COMMENT '科目名称',
            parent_code VARCHAR(20) DEFAULT NULL COMMENT '父科目编码',
            account_type ENUM('资产','负债','所有者权益','成本','损益') NOT NULL COMMENT '科目类型',
            balance_direction ENUM('借','贷') NOT NULL DEFAULT '借' COMMENT '余额方向',
            level INT NOT NULL DEFAULT 1 COMMENT '科目级别',
            is_active TINYINT(1) NOT NULL DEFAULT 1 COMMENT '是否启用',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) COMMENT '会计科目表'
    """, "t_account")

    run_sql(cursor, """
        CREATE TABLE IF NOT EXISTS t_cost_center (
            id INT PRIMARY KEY AUTO_INCREMENT,
            center_code VARCHAR(20) NOT NULL UNIQUE COMMENT '成本中心编码',
            center_name VARCHAR(100) NOT NULL COMMENT '成本中心名称',
            department_id INT DEFAULT NULL COMMENT '关联部门ID',
            manager VARCHAR(50) DEFAULT NULL COMMENT '负责人',
            annual_budget DECIMAL(15,2) NOT NULL DEFAULT 0 COMMENT '年度预算(元)',
            is_active TINYINT(1) NOT NULL DEFAULT 1,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) COMMENT '成本中心表'
    """, "t_cost_center")

    run_sql(cursor, """
        CREATE TABLE IF NOT EXISTS t_journal_entry (
            id INT PRIMARY KEY AUTO_INCREMENT,
            entry_no VARCHAR(30) NOT NULL UNIQUE COMMENT '凭证号',
            entry_date DATE NOT NULL COMMENT '凭证日期',
            entry_type ENUM('收款','付款','转账','期末调整') NOT NULL DEFAULT '转账' COMMENT '凭证类型',
            period VARCHAR(7) NOT NULL COMMENT '会计期间 (YYYY-MM)',
            total_debit DECIMAL(15,2) NOT NULL DEFAULT 0 COMMENT '借方合计',
            total_credit DECIMAL(15,2) NOT NULL DEFAULT 0 COMMENT '贷方合计',
            attachment_count INT NOT NULL DEFAULT 0 COMMENT '附件数',
            source_system VARCHAR(50) DEFAULT NULL COMMENT '来源系统',
            status ENUM('草稿','已审核','已过账','已作废') NOT NULL DEFAULT '草稿' COMMENT '状态',
            prepared_by VARCHAR(50) NOT NULL COMMENT '制单人',
            reviewed_by VARCHAR(50) DEFAULT NULL COMMENT '审核人',
            posted_at DATETIME DEFAULT NULL COMMENT '过账时间',
            memo TEXT COMMENT '备注',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) COMMENT '记账凭证主表'
    """, "t_journal_entry")

    run_sql(cursor, """
        CREATE TABLE IF NOT EXISTS t_journal_item (
            id INT PRIMARY KEY AUTO_INCREMENT,
            entry_id INT NOT NULL COMMENT '凭证ID',
            line_no INT NOT NULL COMMENT '行号',
            account_code VARCHAR(20) NOT NULL COMMENT '科目编码',
            summary VARCHAR(200) NOT NULL COMMENT '摘要',
            debit_amount DECIMAL(15,2) NOT NULL DEFAULT 0 COMMENT '借方金额',
            credit_amount DECIMAL(15,2) NOT NULL DEFAULT 0 COMMENT '贷方金额',
            cost_center_id INT DEFAULT NULL COMMENT '成本中心ID',
            project_code VARCHAR(30) DEFAULT NULL COMMENT '项目编码',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_entry_id (entry_id),
            INDEX idx_account_code (account_code)
        ) COMMENT '凭证分录明细表'
    """, "t_journal_item")

    run_sql(cursor, """
        CREATE TABLE IF NOT EXISTS t_fund_transfer (
            id INT PRIMARY KEY AUTO_INCREMENT,
            transfer_no VARCHAR(30) NOT NULL UNIQUE COMMENT '划转单号',
            transfer_date DATE NOT NULL COMMENT '划转日期',
            transfer_type ENUM('内部调拨','银行转账','现金存取','跨公司划转') NOT NULL COMMENT '划转类型',
            from_account VARCHAR(100) NOT NULL COMMENT '转出账户',
            to_account VARCHAR(100) NOT NULL COMMENT '转入账户',
            amount DECIMAL(15,2) NOT NULL COMMENT '划转金额',
            currency VARCHAR(3) NOT NULL DEFAULT 'CNY' COMMENT '币种',
            exchange_rate DECIMAL(10,4) NOT NULL DEFAULT 1.0000 COMMENT '汇率',
            status ENUM('待审批','已审批','已执行','已拒绝','已撤销') NOT NULL DEFAULT '待审批',
            applicant VARCHAR(50) NOT NULL COMMENT '申请人',
            approver VARCHAR(50) DEFAULT NULL COMMENT '审批人',
            approved_at DATETIME DEFAULT NULL COMMENT '审批时间',
            executed_at DATETIME DEFAULT NULL COMMENT '执行时间',
            purpose VARCHAR(200) COMMENT '用途说明',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) COMMENT '资金划转记录表'
    """, "t_fund_transfer")

    run_sql(cursor, """
        CREATE TABLE IF NOT EXISTS t_budget (
            id INT PRIMARY KEY AUTO_INCREMENT,
            budget_year INT NOT NULL COMMENT '预算年度',
            budget_month INT NOT NULL COMMENT '预算月份',
            cost_center_id INT NOT NULL COMMENT '成本中心ID',
            account_code VARCHAR(20) NOT NULL COMMENT '科目编码',
            budget_amount DECIMAL(15,2) NOT NULL COMMENT '预算金额',
            actual_amount DECIMAL(15,2) NOT NULL DEFAULT 0 COMMENT '实际金额',
            status ENUM('编制中','已审批','执行中','已关闭') NOT NULL DEFAULT '编制中',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uk_budget (budget_year, budget_month, cost_center_id, account_code)
        ) COMMENT '预算管理表'
    """, "t_budget")

    run_sql(cursor, """
        CREATE TABLE IF NOT EXISTS t_invoice (
            id INT PRIMARY KEY AUTO_INCREMENT,
            invoice_no VARCHAR(30) NOT NULL UNIQUE COMMENT '发票号码',
            invoice_code VARCHAR(20) DEFAULT NULL COMMENT '发票代码',
            invoice_type ENUM('增值税专用发票','增值税普通发票','电子发票','收据') NOT NULL COMMENT '发票类型',
            direction ENUM('销项','进项') NOT NULL COMMENT '方向',
            invoice_date DATE NOT NULL COMMENT '开票日期',
            buyer_name VARCHAR(100) NOT NULL COMMENT '购方名称',
            buyer_tax_no VARCHAR(30) DEFAULT NULL COMMENT '购方税号',
            seller_name VARCHAR(100) NOT NULL COMMENT '销方名称',
            seller_tax_no VARCHAR(30) DEFAULT NULL COMMENT '销方税号',
            amount_without_tax DECIMAL(15,2) NOT NULL COMMENT '不含税金额',
            tax_amount DECIMAL(15,2) NOT NULL COMMENT '税额',
            total_amount DECIMAL(15,2) NOT NULL COMMENT '价税合计',
            tax_rate DECIMAL(5,2) NOT NULL COMMENT '税率(%)',
            status ENUM('正常','红冲','作废') NOT NULL DEFAULT '正常',
            verification_status ENUM('未认证','已认证','认证失败') DEFAULT NULL COMMENT '认证状态',
            related_entry_id INT DEFAULT NULL COMMENT '关联凭证ID',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_invoice_date (invoice_date),
            INDEX idx_seller (seller_name)
        ) COMMENT '发票管理表'
    """, "t_invoice")

    run_sql(cursor, """
        CREATE TABLE IF NOT EXISTS t_receivable_payable (
            id INT PRIMARY KEY AUTO_INCREMENT,
            rp_type ENUM('应收','应付') NOT NULL COMMENT '类型',
            rp_no VARCHAR(30) NOT NULL UNIQUE COMMENT '单据号',
            counterparty VARCHAR(100) NOT NULL COMMENT '往来单位',
            contract_no VARCHAR(30) DEFAULT NULL COMMENT '合同号',
            original_amount DECIMAL(15,2) NOT NULL COMMENT '原始金额',
            settled_amount DECIMAL(15,2) NOT NULL DEFAULT 0 COMMENT '已结金额',
            currency VARCHAR(3) NOT NULL DEFAULT 'CNY',
            due_date DATE NOT NULL COMMENT '到期日',
            status ENUM('未结','部分结清','已结清','逾期','核销') NOT NULL DEFAULT '未结',
            related_invoice_id INT DEFAULT NULL COMMENT '关联发票ID',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_due_date (due_date),
            INDEX idx_counterparty (counterparty),
            INDEX idx_status (status)
        ) COMMENT '应收应付表'
    """, "t_receivable_payable")

    run_sql(cursor, """
        CREATE TABLE IF NOT EXISTS t_expense_claim (
            id INT PRIMARY KEY AUTO_INCREMENT,
            claim_no VARCHAR(30) NOT NULL UNIQUE COMMENT '报销单号',
            claim_date DATE NOT NULL COMMENT '报销日期',
            claimant VARCHAR(50) NOT NULL COMMENT '报销人',
            department_id INT DEFAULT NULL COMMENT '部门ID',
            cost_center_id INT DEFAULT NULL COMMENT '成本中心ID',
            expense_type ENUM('差旅','交通','餐饮','办公','招待','培训','其他') NOT NULL COMMENT '费用类型',
            total_amount DECIMAL(15,2) NOT NULL COMMENT '报销总额',
            approved_amount DECIMAL(15,2) DEFAULT NULL COMMENT '审批金额',
            status ENUM('草稿','已提交','已审批','已付款','已拒绝','已撤回') NOT NULL DEFAULT '草稿',
            approver VARCHAR(50) DEFAULT NULL COMMENT '审批人',
            approved_at DATETIME DEFAULT NULL,
            paid_at DATETIME DEFAULT NULL,
            description TEXT COMMENT '费用说明',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_claimant (claimant),
            INDEX idx_claim_date (claim_date)
        ) COMMENT '费用报销表'
    """, "t_expense_claim")

    run_sql(cursor, """
        CREATE TABLE IF NOT EXISTS t_fixed_asset (
            id INT PRIMARY KEY AUTO_INCREMENT,
            asset_code VARCHAR(30) NOT NULL UNIQUE COMMENT '资产编码',
            asset_name VARCHAR(100) NOT NULL COMMENT '资产名称',
            asset_category ENUM('房屋建筑','机器设备','运输工具','电子设备','办公家具','其他') NOT NULL COMMENT '资产类别',
            acquisition_date DATE NOT NULL COMMENT '购入日期',
            acquisition_cost DECIMAL(15,2) NOT NULL COMMENT '原值',
            salvage_value DECIMAL(15,2) NOT NULL DEFAULT 0 COMMENT '残值',
            useful_life_months INT NOT NULL COMMENT '使用月数',
            monthly_depreciation DECIMAL(15,2) NOT NULL COMMENT '月折旧额',
            accumulated_depreciation DECIMAL(15,2) NOT NULL DEFAULT 0 COMMENT '累计折旧',
            depreciation_method ENUM('直线法','双倍余额递减法','年数总和法') NOT NULL DEFAULT '直线法',
            location VARCHAR(100) COMMENT '存放地点',
            custodian VARCHAR(50) COMMENT '保管人',
            status ENUM('在用','闲置','已报废','已处置') NOT NULL DEFAULT '在用',
            cost_center_id INT DEFAULT NULL COMMENT '成本中心ID',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        ) COMMENT '固定资产表'
    """, "t_fixed_asset")

    print("All tables created.\n")


def seed_data(conn, cursor):
    print("=== Seeding data ===")
    random.seed(42)
    today = date.today()
    months = _recent_months(6)

    # -- 会计科目 (29 条) --
    accounts = [
        ("1001", "库存现金", None, "资产", "借", 1),
        ("1002", "银行存款", None, "资产", "借", 1),
        ("100201", "银行存款-工行", "1002", "资产", "借", 2),
        ("100202", "银行存款-建行", "1002", "资产", "借", 2),
        ("100203", "银行存款-招行", "1002", "资产", "借", 2),
        ("1122", "应收账款", None, "资产", "借", 1),
        ("1221", "其他应收款", None, "资产", "借", 1),
        ("1403", "原材料", None, "资产", "借", 1),
        ("1601", "固定资产", None, "资产", "借", 1),
        ("1602", "累计折旧", None, "资产", "贷", 1),
        ("2202", "应付账款", None, "负债", "贷", 1),
        ("2211", "应付职工薪酬", None, "负债", "贷", 1),
        ("2221", "应交税费", None, "负债", "贷", 1),
        ("222101", "应交增值税", "2221", "负债", "贷", 2),
        ("222102", "应交企业所得税", "2221", "负债", "贷", 2),
        ("2241", "其他应付款", None, "负债", "贷", 1),
        ("4001", "实收资本", None, "所有者权益", "贷", 1),
        ("4101", "盈余公积", None, "所有者权益", "贷", 1),
        ("4104", "利润分配", None, "所有者权益", "贷", 1),
        ("5001", "生产成本", None, "成本", "借", 1),
        ("5401", "管理费用", None, "损益", "借", 1),
        ("540101", "管理费用-工资", "5401", "损益", "借", 2),
        ("540102", "管理费用-折旧", "5401", "损益", "借", 2),
        ("540103", "管理费用-办公", "5401", "损益", "借", 2),
        ("5402", "销售费用", None, "损益", "借", 1),
        ("5403", "财务费用", None, "损益", "借", 1),
        ("6001", "主营业务收入", None, "损益", "贷", 1),
        ("6051", "其他业务收入", None, "损益", "贷", 1),
        ("6401", "主营业务成本", None, "损益", "借", 1),
    ]
    for a in accounts:
        parent = "NULL" if a[2] is None else f"'{a[2]}'"
        run_sql(cursor,
            f"INSERT IGNORE INTO t_account (account_code, account_name, parent_code, account_type, balance_direction, level) "
            f"VALUES ('{a[0]}', '{a[1]}', {parent}, '{a[3]}', '{a[4]}', {a[5]})",
            f"account {a[0]}")
    conn.commit()

    # -- 成本中心 (8 条) --
    cost_centers = [
        ("CC001", "总裁办", 1, "张总", 500000),
        ("CC002", "财务部", 2, "李经理", 300000),
        ("CC003", "研发部", 3, "王总监", 2000000),
        ("CC004", "销售部", 4, "赵经理", 1500000),
        ("CC005", "人力资源部", 5, "刘经理", 400000),
        ("CC006", "生产部", 6, "陈主任", 3000000),
        ("CC007", "采购部", 7, "周经理", 800000),
        ("CC008", "IT部", 8, "吴工", 600000),
    ]
    for cc in cost_centers:
        run_sql(cursor,
            f"INSERT IGNORE INTO t_cost_center (center_code, center_name, department_id, manager, annual_budget) "
            f"VALUES ('{cc[0]}', '{cc[1]}', {cc[2]}, '{cc[3]}', {cc[4]})",
            f"cost_center {cc[0]}")
    conn.commit()

    # -- 记账凭证 + 分录 (每个凭证 10 条) --
    entry_no = 1
    summaries = [
        "支付供应商货款", "收到客户回款", "报销差旅费", "计提折旧",
        "支付员工工资", "缴纳社保", "采购原材料", "销售商品收款",
        "支付租金", "收到投资款", "支付水电费", "报销招待费",
        "银行手续费", "利息收入", "缴纳税款",
    ]
    reviewers = ["赵主管", "钱经理"]
    preparers = ["张会计", "李出纳", "王会计"]

    for y, m in months:
        ym = f"{y}-{m:02d}"
        for _ in range(10):
            entry_date = _rand_date(y, m)
            entry_type = random.choice(["收款", "付款", "转账", "转账", "转账"])
            status = random.choice(["已审核", "已过账", "已过账", "已过账"])
            prepared = random.choice(preparers)
            reviewer = random.choice(reviewers)

            num_lines = random.randint(2, 5)
            total_debit = 0.0
            total_credit = 0.0
            items = []

            for line_no in range(1, num_lines + 1):
                if line_no < num_lines:
                    amount = round(random.uniform(1000, 500000), 2)
                    is_debit = random.choice([True, False])
                    debit = amount if is_debit else 0
                    credit = 0 if is_debit else amount
                else:
                    debit = round(total_credit - total_debit, 2) if total_credit > total_debit else 0
                    credit = round(total_debit - total_credit, 2) if total_debit > total_credit else 0

                total_debit += debit
                total_credit += credit

                acc = random.choice(accounts[:15])
                summary = random.choice(summaries)
                cc_num = random.randint(1, 8)

                items.append((line_no, acc[0], summary, debit, credit, cc_num))

            entry_no_str = f"PZ-{y}{m:02d}-{entry_no:04d}"
            posted_at = f"'{entry_date} 10:00:00'" if status == "已过账" else "NULL"
            run_sql(cursor,
                f"INSERT IGNORE INTO t_journal_entry (entry_no, entry_date, entry_type, period, "
                f"total_debit, total_credit, attachment_count, status, prepared_by, reviewed_by, posted_at, memo) "
                f"VALUES ('{entry_no_str}', '{entry_date}', '{entry_type}', '{ym}', "
                f"{total_debit:.2f}, {total_credit:.2f}, {random.randint(1, 5)}, '{status}', "
                f"'{prepared}', '{reviewer}', {posted_at}, '自动生成')",
                f"entry {entry_no_str}")

            cursor.execute("SELECT LAST_INSERT_ID()")
            entry_id = cursor.fetchone()[0]

            for item in items:
                run_sql(cursor,
                    f"INSERT IGNORE INTO t_journal_item (entry_id, line_no, account_code, summary, "
                    f"debit_amount, credit_amount, cost_center_id, project_code) "
                    f"VALUES ({entry_id}, {item[0]}, '{item[1]}', '{item[2]}', "
                    f"{item[3]:.2f}, {item[4]:.2f}, {item[5]}, NULL)",
                    "")

            entry_no += 1
    conn.commit()
    print(f"  [OK] {len(months) * 10} journal entries with items")

    seed_prior_year_loss_scenario(conn, cursor)

    # -- 资金划转 (每月 5 条) --
    ft_no = 1
    transfer_purposes = [
        "日常资金调拨", "支付供应商货款", "缴纳税款", "发放工资",
        "归还贷款", "投资理财", "备用金提取", "跨账户归集",
    ]
    for y, m in months:
        for _ in range(5):
            transfer_no = f"FT-{y}{m:02d}-{ft_no:04d}"
            transfer_date = _rand_date(y, m)
            t_type = random.choice(["内部调拨", "银行转账", "银行转账", "跨公司划转"])
            from_acc = random.choice(["工行基本户", "建行一般户", "招行一般户", "现金库"])
            to_acc = random.choice(["工行基本户", "建行一般户", "招行一般户", "现金库"])
            amount = round(random.uniform(10000, 2000000), 2)
            status = random.choice(["已审批", "已执行", "已执行", "已执行"])
            applicant = random.choice(["张会计", "李出纳", "王经理"])
            approver = random.choice(["赵主管", "钱经理"])
            purpose = random.choice(transfer_purposes)
            approved_at = f"'{transfer_date} 14:00:00'" if status != "待审批" else "NULL"
            executed_at = f"'{transfer_date} 16:00:00'" if status == "已执行" else "NULL"
            run_sql(cursor,
                f"INSERT IGNORE INTO t_fund_transfer (transfer_no, transfer_date, transfer_type, "
                f"from_account, to_account, amount, status, applicant, approver, approved_at, executed_at, purpose) "
                f"VALUES ('{transfer_no}', '{transfer_date}', '{t_type}', '{from_acc}', '{to_acc}', "
                f"{amount:.2f}, '{status}', '{applicant}', '{approver}', {approved_at}, {executed_at}, '{purpose}')",
                f"transfer {transfer_no}")
            ft_no += 1
    conn.commit()
    print(f"  [OK] {len(months) * 5} fund transfers")

    # -- 预算 (每月 x 6 中心 x 5 科目) --
    budget_accounts = ["540101", "540102", "540103", "5402", "5403"]
    for y, m in months:
        for cc_id in range(1, 7):
            for acc_code in budget_accounts:
                budget = round(random.uniform(10000, 200000), 2)
                # Past months have actual, current month partial
                if (y, m) < (today.year, today.month):
                    actual = round(budget * random.uniform(0.6, 1.3), 2)
                    status = "执行中"
                elif (y, m) == (today.year, today.month):
                    actual = round(budget * random.uniform(0.1, 0.5), 2)
                    status = "执行中"
                else:
                    actual = 0
                    status = "编制中"
                run_sql(cursor,
                    f"INSERT IGNORE INTO t_budget (budget_year, budget_month, cost_center_id, "
                    f"account_code, budget_amount, actual_amount, status) "
                    f"VALUES ({y}, {m}, {cc_id}, '{acc_code}', {budget:.2f}, {actual:.2f}, '{status}')",
                    "")
    conn.commit()
    print(f"  [OK] {len(months) * 30} budget records")

    # -- 发票 (每月 ~8 条) --
    buyers = ["北京科技有限公司", "上海贸易有限公司", "深圳电子有限公司", "广州制造有限公司"]
    sellers = ["原材料供应商A", "设备供应商B", "办公用品供应商C", "物流公司D"]
    inv_no = 1
    for y, m in months:
        for _ in range(8):
            invoice_no = f"INV-{y}{m:02d}-{inv_no:06d}"
            direction = random.choice(["销项", "进项"])
            inv_date = _rand_date(y, m)
            inv_type = random.choice(["增值税专用发票", "增值税普通发票", "电子发票"])
            tax_rate = random.choice([13, 9, 6, 3])
            amount = round(random.uniform(1000, 300000), 2)
            tax = round(amount * tax_rate / 100, 2)
            total = round(amount + tax, 2)
            if direction == "销项":
                buyer = random.choice(buyers)
                seller = "本公司"
            else:
                buyer = "本公司"
                seller = random.choice(sellers)
            status = random.choice(["正常", "正常", "正常", "红冲"])
            ver_status = random.choice(["已认证", "已认证", "未认证"]) if direction == "进项" else None
            vs = "NULL" if ver_status is None else f"'{ver_status}'"
            run_sql(cursor,
                f"INSERT IGNORE INTO t_invoice (invoice_no, invoice_type, direction, invoice_date, "
                f"buyer_name, seller_name, amount_without_tax, tax_amount, total_amount, tax_rate, "
                f"status, verification_status) "
                f"VALUES ('{invoice_no}', '{inv_type}', '{direction}', '{inv_date}', "
                f"'{buyer}', '{seller}', {amount:.2f}, {tax:.2f}, {total:.2f}, {tax_rate}, "
                f"'{status}', {vs})",
                "")
            inv_no += 1
    conn.commit()
    print(f"  [OK] {len(months) * 8} invoices")

    # -- 应收应付 (每月 ~7 条) --
    counterparties = ["北京科技有限公司", "上海贸易有限公司", "深圳电子有限公司",
                      "原材料供应商A", "设备供应商B", "物流公司D"]
    rp_no = 1
    for y, m in months:
        for _ in range(7):
            rp_type = random.choice(["应收", "应付"])
            rp_no_str = f"{'AR' if rp_type == '应收' else 'AP'}-{y}{m:02d}-{rp_no:04d}"
            cp = random.choice(counterparties)
            amount = round(random.uniform(5000, 500000), 2)
            roll = random.random()
            if roll < 0.3:
                settled = amount
                status = "已结清"
                due_date = _rand_date(y, m)
            elif roll < 0.5:
                settled = round(amount * random.uniform(0.3, 0.9), 2)
                status = "部分结清"
                due_date = _rand_date(y, m)
            elif roll < 0.8:
                settled = 0
                status = "未结"
                # Future due date
                future_m = m + random.randint(1, 3)
                future_y = y
                if future_m > 12:
                    future_m -= 12
                    future_y += 1
                due_date = _rand_date(future_y, future_m)
            else:
                settled = round(amount * random.uniform(0, 0.5), 2)
                status = "逾期"
                # Past due date
                past_m = m - random.randint(1, 2)
                past_y = y
                if past_m <= 0:
                    past_m += 12
                    past_y -= 1
                due_date = _rand_date(past_y, past_m)
            run_sql(cursor,
                f"INSERT IGNORE INTO t_receivable_payable (rp_type, rp_no, counterparty, original_amount, "
                f"settled_amount, due_date, status) "
                f"VALUES ('{rp_type}', '{rp_no_str}', '{cp}', {amount:.2f}, {settled:.2f}, '{due_date}', '{status}')",
                "")
            rp_no += 1
    conn.commit()
    print(f"  [OK] {len(months) * 7} receivable/payable records")

    # -- 费用报销 (每月 ~7 条) --
    claimants = [
        ("张三", 3), ("李四", 4), ("王五", 3), ("赵六", 2), ("钱七", 5),
        ("孙八", 6), ("周九", 7), ("吴十", 8),
    ]
    expense_descs = [
        "出差北京客户拜访", "团队建设聚餐", "购买办公文具", "项目培训费用",
        "商务招待餐费", "市内交通费", "打印复印费", "软件许可费",
        "服务器托管费", "差旅机票费用", "客户礼品采购", "办公家具购置",
    ]
    approvers_list = ["赵主管", "钱经理", "王总监"]
    claim_no = 1
    for y, m in months:
        for _ in range(7):
            claim_no_str = f"EXP-{y}{m:02d}-{claim_no:04d}"
            claim_date = _rand_date(y, m)
            claimant, cc_id = random.choice(claimants)
            exp_type = random.choice(["差旅", "交通", "餐饮", "办公", "招待", "培训"])
            amount = round(random.uniform(200, 30000), 2)
            approved = round(amount * random.uniform(0.85, 1.0), 2)
            approver = random.choice(approvers_list)
            # Past months: mostly approved/paid; current month: mix
            if (y, m) < (today.year, today.month):
                status = random.choice(["已审批", "已付款", "已付款", "已付款", "已拒绝"])
            else:
                status = random.choice(["草稿", "已提交", "已审批", "已拒绝"])
            approved_at = f"'{claim_date}'" if status in ("已审批", "已付款") else "NULL"
            paid_at = f"'{claim_date}'" if status == "已付款" else "NULL"
            desc = random.choice(expense_descs)
            run_sql(cursor,
                f"INSERT IGNORE INTO t_expense_claim (claim_no, claim_date, claimant, cost_center_id, "
                f"expense_type, total_amount, approved_amount, status, approver, approved_at, paid_at, description) "
                f"VALUES ('{claim_no_str}', '{claim_date}', '{claimant}', {cc_id}, "
                f"'{exp_type}', {amount:.2f}, {approved:.2f}, '{status}', "
                f"'{approver}', {approved_at}, {paid_at}, '{desc}')",
                "")
            claim_no += 1
    conn.commit()
    print(f"  [OK] {len(months) * 7} expense claims")

    # -- 固定资产 (8 条) --
    assets = [
        ("FA-001", "办公楼A栋", "房屋建筑", "2020-01-15", 5000000, 50000, 360, "总部大楼", "行政部", 1),
        ("FA-002", "CNC数控机床", "机器设备", "2022-06-01", 1200000, 60000, 120, "生产车间A", "生产部", 6),
        ("FA-003", "公司班车", "运输工具", "2023-03-15", 350000, 35000, 60, "公司车库", "行政部", 1),
        ("FA-004", "服务器集群", "电子设备", "2023-09-01", 280000, 14000, 36, "机房", "IT部", 8),
        ("FA-005", "投影仪", "电子设备", "2024-01-10", 15000, 750, 36, "会议室", "行政部", 1),
        ("FA-006", "办公桌椅套装", "办公家具", "2024-02-01", 8000, 400, 60, "办公区", "行政部", 1),
        ("FA-007", "3D打印机", "机器设备", "2024-06-15", 180000, 9000, 48, "研发中心", "研发部", 3),
        ("FA-008", "空调系统", "其他", "2021-05-01", 200000, 10000, 120, "全楼", "行政部", 1),
    ]
    for a in assets:
        monthly_dep = round((a[4] - a[5]) / a[6], 2)
        acq_date = date.fromisoformat(a[3])
        months_used = min((today.year - acq_date.year) * 12 + today.month - acq_date.month, a[6])
        months_used = max(months_used, 1)
        accum_dep = round(monthly_dep * months_used, 2)
        run_sql(cursor,
            f"INSERT IGNORE INTO t_fixed_asset (asset_code, asset_name, asset_category, acquisition_date, "
            f"acquisition_cost, salvage_value, useful_life_months, monthly_depreciation, "
            f"accumulated_depreciation, location, custodian, cost_center_id) "
            f"VALUES ('{a[0]}', '{a[1]}', '{a[2]}', '{a[3]}', {a[4]}, {a[5]}, {a[6]}, "
            f"{monthly_dep}, {accum_dep}, '{a[7]}', '{a[8]}', {a[9]})",
            f"asset {a[0]}")
    conn.commit()
    print("  [OK] 8 fixed assets")
    print()


def main():
    conn = get_conn()
    cursor = conn.cursor()
    try:
        create_tables(cursor)
        seed_data(conn, cursor)
    finally:
        cursor.close()
        conn.close()

    print("=== All done! ===")


if __name__ == "__main__":
    main()
