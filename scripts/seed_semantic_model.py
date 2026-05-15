"""Seed semantic model (field-level business mappings) into MySQL.

Column names must match the actual table schemas created by seed_financial.py.

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
        cursorclass=pymysql.cursors.DictCursor,
    )


_TABLE_VISIBLE_SEMANTICS = {
    "t_account": "会计科目表，包含科目编码、科目名称、科目类型、余额方向和启用状态",
    "t_cost_center": "成本中心/部门预算责任中心表，包含成本中心名称、部门映射、负责人和年度预算",
    "t_journal_entry": "记账凭证主表，包含凭证号、凭证日期、会计期间、凭证状态、制单人和审核人",
    "t_journal_item": "凭证分录明细表，包含科目、摘要、借方金额、贷方金额、成本中心和项目编码",
    "t_budget": "预算管理表，包含预算年度、预算月份、成本中心、会计科目、预算金额、实际金额和审批状态",
    "t_invoice": "发票管理表，包含发票号码、发票方向、开票日期、购销方、税额、价税合计、认证状态和关联凭证",
    "t_receivable_payable": "应收应付表，包含往来单位、原始金额、已结金额、状态、到期日、关联发票和核销凭证",
    "t_expense_claim": "费用报销表，包含报销人、部门、成本中心、费用类型、报销总额、审批金额和报销状态",
    "t_fixed_asset": "固定资产表，包含资产名称、资产类别、购入日期、原值、折旧、存放地点和成本中心",
    "t_fund_transfer": "资金划转记录表，包含划转单号、划转日期、转出账户、转入账户、金额、申请人和审批状态",
    "t_user": "用户/员工账号信息表，包含真实姓名、联系电话、邮箱、注册时间、账号状态",
    "t_role": "系统角色信息表，包含角色名称、角色编码、角色状态和创建时间",
    "t_user_role": "用户角色绑定关系表，关联用户与系统角色，用于查询用户拥有哪些角色",
    "t_department": "组织部门信息表，包含部门名称、上级部门、部门负责人、联系电话和状态",
    "t_user_department": "用户部门归属关系表，关联用户与部门，并标识是否部门负责人",
}


def table_visible_semantics() -> dict[str, str]:
    """Return table-level business descriptions used by select_tables."""
    return dict(_TABLE_VISIBLE_SEMANTICS)


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS t_semantic_model (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                table_name VARCHAR(128) NOT NULL COMMENT '物理表名',
                column_name VARCHAR(128) NOT NULL COMMENT '物理字段名',
                column_type VARCHAR(128) COMMENT '字段类型（如 varchar(64), decimal(20,2)）',
                column_comment VARCHAR(512) COMMENT '字段注释（来自 information_schema）',
                is_pk TINYINT DEFAULT 0 COMMENT '是否主键',
                is_fk TINYINT DEFAULT 0 COMMENT '是否外键',
                ref_table VARCHAR(128) COMMENT '外键引用表',
                ref_column VARCHAR(128) COMMENT '外键引用字段',
                business_name VARCHAR(256) COMMENT '业务名称',
                synonyms TEXT COMMENT '同义词，逗号分隔',
                business_description TEXT COMMENT '业务描述（枚举值、状态码、计算逻辑等）',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                UNIQUE KEY uk_table_col (table_name, column_name)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='语义模型：字段级业务映射+技术schema'
        """)
        # 兼容旧表：添加新列
        for col, col_def in [
            ("column_type", "VARCHAR(128) COMMENT '字段类型'"),
            ("column_comment", "VARCHAR(512) COMMENT '字段注释'"),
            ("is_pk", "TINYINT DEFAULT 0 COMMENT '是否主键'"),
            ("is_fk", "TINYINT DEFAULT 0 COMMENT '是否外键'"),
            ("ref_table", "VARCHAR(128) COMMENT '外键引用表'"),
            ("ref_column", "VARCHAR(128) COMMENT '外键引用字段'"),
        ]:
            try:
                cur.execute(f"ALTER TABLE t_semantic_model ADD COLUMN {col} {col_def}")
            except Exception:
                pass  # 列已存在
    conn.commit()
    print("t_semantic_model table created/updated")


def sync_schema_from_information_schema(conn):
    """从 information_schema 自动同步字段类型、注释、主键、外键信息。"""
    db = settings.mysql.database

    def _lower_keys(row: dict) -> dict:
        return {k.lower(): v for k, v in row.items()}

    # 1. 获取所有表的字段信息
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name, column_type, column_comment, column_key "
            "FROM information_schema.columns "
            "WHERE table_schema = %s "
            "ORDER BY table_name, ordinal_position",
            (db,),
        )
        columns = [_lower_keys(r) for r in cur.fetchall()]

    # 2. 获取外键信息
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name, referenced_table_name, referenced_column_name "
            "FROM information_schema.key_column_usage "
            "WHERE table_schema = %s AND referenced_table_name IS NOT NULL",
            (db,),
        )
        fk_rows = [_lower_keys(r) for r in cur.fetchall()]
    fk_map = {}
    for r in fk_rows:
        fk_map[(r["table_name"], r["column_name"])] = (
            r["referenced_table_name"], r["referenced_column_name"]
        )

    # 3. 批量更新 t_semantic_model
    updated = 0
    with conn.cursor() as cur:
        for col in columns:
            tbl, col_name = col["table_name"], col["column_name"]
            is_pk = 1 if col.get("column_key") == "PRI" else 0
            ref_tbl, ref_col = fk_map.get((tbl, col_name), (None, None))
            is_fk = 1 if ref_tbl else 0
            cur.execute(
                """INSERT INTO t_semantic_model (table_name, column_name, column_type, column_comment, is_pk, is_fk, ref_table, ref_column)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                       column_type=VALUES(column_type),
                       column_comment=VALUES(column_comment),
                       is_pk=VALUES(is_pk),
                       is_fk=VALUES(is_fk),
                       ref_table=VALUES(ref_table),
                       ref_column=VALUES(ref_column)""",
                (tbl, col_name, col.get("column_type"), col.get("column_comment"),
                 is_pk, is_fk, ref_tbl, ref_col),
            )
            updated += 1
    conn.commit()
    print(f"Synced {updated} columns from information_schema")


def semantic_model_records() -> list[tuple[str, str, str, str, str]]:
    """Return field-level semantic model seed records.

    Column names must match the actual table schemas. Records are metadata, not
    runtime routing rules; select_tables remains driven by database semantics.
    """
    return [
        # t_account
        ("t_account", "account_code", "科目编码", "会计科目代码", "财务科目唯一编码，如 1001=库存现金"),
        ("t_account", "account_name", "科目名称", "会计科目, 账户名称", "财务科目名称"),
        ("t_account", "account_type", "科目类型", "账户类型", "资产/负债/所有者权益/成本/损益"),
        ("t_account", "balance_direction", "余额方向", "借贷方向", "借/贷"),
        ("t_account", "level", "科目层级", "", "1=一级科目, 2=二级科目"),
        ("t_account", "parent_code", "父科目编码", "上级科目", "用于多级科目树形结构"),
        ("t_account", "is_active", "是否启用", "", "1=启用, 0=停用"),

        # t_cost_center
        ("t_cost_center", "center_code", "成本中心编码", "部门编码", "成本中心唯一编码"),
        ("t_cost_center", "center_name", "成本中心名称", "部门名称", "如：研发部、市场部、财务部"),
        ("t_cost_center", "manager", "负责人", "部门经理, 主管", "成本中心负责人姓名"),
        ("t_cost_center", "annual_budget", "年度预算", "全年预算", "该成本中心的年度预算金额"),

        # organization / access management tables
        ("t_user", "username", "用户名", "登录名, 账号, 用户账号", "系统登录账号"),
        ("t_user", "real_name", "真实姓名", "员工姓名, 用户姓名, 姓名", "用户或员工的真实姓名"),
        ("t_user", "gender", "性别", "", "用户性别编码"),
        ("t_user", "email", "邮箱地址", "电子邮箱, 邮箱", "用户联系邮箱"),
        ("t_user", "phone", "联系电话", "手机号, 电话, 联系方式", "用户联系电话"),
        ("t_user", "register_time", "注册时间", "创建时间, 入库时间", "用户账号注册时间"),
        ("t_user", "status", "用户状态", "账号状态, 是否正常", "0=停用, 1=正常"),

        ("t_role", "name", "角色名称", "系统角色, 权限角色", "系统角色的中文名称"),
        ("t_role", "code", "角色编码", "角色代码", "系统角色唯一编码"),
        ("t_role", "description", "角色描述", "角色说明", "角色职责或权限说明"),
        ("t_role", "status", "角色状态", "是否正常, 是否启用", "0=停用, 1=正常"),
        ("t_role", "created_at", "角色创建时间", "创建时间", "角色记录创建时间"),

        ("t_user_role", "user_id", "用户ID", "员工ID, 用户编号", "关联 t_user.id"),
        ("t_user_role", "role_id", "角色ID", "系统角色ID", "关联 t_role.id"),
        ("t_user_role", "created_at", "绑定时间", "分配时间, 创建时间", "用户角色绑定关系创建时间"),

        ("t_department", "name", "部门名称", "组织名称, 部门", "组织部门名称"),
        ("t_department", "parent_id", "上级部门ID", "父部门, 上级组织", "关联 t_department.id，表示部门层级"),
        ("t_department", "manager", "部门负责人", "负责人, 部门经理, 主管", "部门负责人姓名"),
        ("t_department", "phone", "联系电话", "部门电话, 联系方式", "部门联系电话"),
        ("t_department", "status", "部门状态", "是否正常, 是否启用", "0=停用, 1=正常"),
        ("t_department", "created_at", "部门创建时间", "创建时间", "部门记录创建时间"),

        ("t_user_department", "user_id", "用户ID", "员工ID, 用户编号", "关联 t_user.id"),
        ("t_user_department", "department_id", "部门ID", "组织ID", "关联 t_department.id"),
        ("t_user_department", "is_leader", "是否部门负责人", "是否负责人, 是否主管", "1=是部门负责人, 0=不是部门负责人"),
        ("t_user_department", "created_at", "归属创建时间", "分配时间, 创建时间", "用户部门归属关系创建时间"),

        # t_journal_entry
        ("t_journal_entry", "entry_no", "凭证号", "记账凭证编号", "会计凭证唯一编号"),
        ("t_journal_entry", "entry_date", "凭证日期", "记账日期", "凭证录入日期"),
        ("t_journal_entry", "entry_type", "凭证类型", "", "收款/付款/转账/期末调整"),
        ("t_journal_entry", "period", "会计期间", "账期", "格式 YYYY-MM，如 2025-01"),
        ("t_journal_entry", "total_debit", "借方合计", "借方总额", "凭证借方金额合计"),
        ("t_journal_entry", "total_credit", "贷方合计", "贷方总额", "凭证贷方金额合计"),
        ("t_journal_entry", "status", "凭证状态", "", "草稿/已审核/已过账/已作废"),
        ("t_journal_entry", "prepared_by", "制单人", "录入人, 制单会计", "凭证制单人姓名"),
        ("t_journal_entry", "reviewed_by", "审核人", "审核会计", "凭证审核人姓名"),
        ("t_journal_entry", "attachment_count", "附件数", "", "凭证附件张数"),
        ("t_journal_entry", "source_system", "来源系统", "", "凭证来源的业务系统"),

        # t_journal_item
        ("t_journal_item", "entry_id", "凭证ID", "", "关联 t_journal_entry.id"),
        ("t_journal_item", "line_no", "行号", "", "分录行号"),
        ("t_journal_item", "account_code", "科目编码", "会计科目", "关联 t_account.account_code"),
        ("t_journal_item", "summary", "摘要", "备注, 说明", "凭证行摘要说明"),
        ("t_journal_item", "debit_amount", "借方金额", "借记金额", "凭证行借方金额"),
        ("t_journal_item", "credit_amount", "贷方金额", "贷记金额", "凭证行贷方金额"),
        ("t_journal_item", "cost_center_id", "成本中心ID", "部门", "关联 t_cost_center.id"),
        ("t_journal_item", "project_code", "项目编码", "", "项目核算编码"),

        # t_fund_transfer
        ("t_fund_transfer", "transfer_no", "转账单号", "划款单号", "资金划转单号"),
        ("t_fund_transfer", "transfer_date", "划转日期", "转账日期", "资金划转日期"),
        ("t_fund_transfer", "transfer_type", "转账类型", "划款类型", "内部调拨/银行转账/现金存取/跨公司划转"),
        ("t_fund_transfer", "from_account", "转出账户", "付款账户", "资金转出的银行账户"),
        ("t_fund_transfer", "to_account", "转入账户", "收款账户", "资金转入的银行账户"),
        ("t_fund_transfer", "amount", "划转金额", "转账金额", "资金划转金额"),
        ("t_fund_transfer", "currency", "币种", "", "CNY/USD/EUR"),
        ("t_fund_transfer", "status", "转账状态", "", "待审批/已审批/已执行/已拒绝/已撤销"),
        ("t_fund_transfer", "applicant", "申请人", "", "资金划转申请人"),
        ("t_fund_transfer", "approver", "审批人", "", "资金划转审批人"),
        ("t_fund_transfer", "purpose", "用途说明", "", "资金划转用途"),

        # t_budget
        ("t_budget", "budget_year", "预算年度", "", "预算所属年份"),
        ("t_budget", "budget_month", "预算月份", "", "预算所属月份，1-12"),
        ("t_budget", "cost_center_id", "成本中心ID", "部门", "关联 t_cost_center.id"),
        ("t_budget", "account_code", "科目编码", "", "关联 t_account.account_code"),
        ("t_budget", "budget_amount", "预算金额", "预算额度", "该月预算金额"),
        ("t_budget", "actual_amount", "实际金额", "实际发生额", "该月实际发生金额"),
        ("t_budget", "status", "预算状态", "", "编制中/已审批/执行中/已关闭"),

        # t_invoice
        ("t_invoice", "invoice_no", "发票号码", "", "发票唯一号码"),
        ("t_invoice", "invoice_type", "发票类型", "", "增值税专用发票/增值税普通发票/电子发票/收据"),
        ("t_invoice", "direction", "开票方向", "", "销项=对外开票, 进项=收到发票"),
        ("t_invoice", "invoice_date", "开票日期", "", "发票开具日期"),
        ("t_invoice", "buyer_name", "购方名称", "购买方", "发票购买方名称"),
        ("t_invoice", "seller_name", "销方名称", "销售方", "发票销售方名称"),
        ("t_invoice", "amount_without_tax", "不含税金额", "发票金额", "发票不含税金额"),
        ("t_invoice", "tax_amount", "税额", "", "发票税额"),
        ("t_invoice", "total_amount", "价税合计", "含税金额", "不含税金额 + 税额"),
        ("t_invoice", "tax_rate", "税率", "", "增值税税率(%)"),
        ("t_invoice", "status", "发票状态", "", "正常/红冲/作废"),
        ("t_invoice", "verification_status", "认证状态", "", "进项发票: 未认证/已认证/认证失败"),

        # t_receivable_payable
        ("t_receivable_payable", "rp_type", "类型", "", "应收/应付"),
        ("t_receivable_payable", "rp_no", "单据号", "", "应收应付单据编号"),
        ("t_receivable_payable", "counterparty", "往来单位", "对方单位, 客户/供应商", "交易对手方名称"),
        ("t_receivable_payable", "contract_no", "合同号", "", "关联合同编号"),
        ("t_receivable_payable", "original_amount", "原始金额", "合同金额", "应收/应付原始金额"),
        ("t_receivable_payable", "settled_amount", "已结金额", "已收/已付金额", "已结算的金额"),
        ("t_receivable_payable", "due_date", "到期日", "", "应收/应付到期日期"),
        ("t_receivable_payable", "status", "状态", "", "未结/部分结清/已结清/逾期/核销"),

        # t_expense_claim
        ("t_expense_claim", "claim_no", "报销单号", "", "报销单编号"),
        ("t_expense_claim", "claim_date", "报销日期", "", "报销申请日期"),
        ("t_expense_claim", "claimant", "报销人", "申请人", "报销申请人姓名"),
        ("t_expense_claim", "cost_center_id", "成本中心ID", "部门", "关联 t_cost_center.id"),
        ("t_expense_claim", "expense_type", "费用类型", "报销类型", "差旅/交通/餐饮/办公/招待/培训/其他"),
        ("t_expense_claim", "total_amount", "报销总额", "费用金额", "报销总金额"),
        ("t_expense_claim", "approved_amount", "审批金额", "", "审批通过的金额"),
        ("t_expense_claim", "status", "报销状态", "", "草稿/已提交/已审批/已付款/已拒绝/已撤回"),
        ("t_expense_claim", "approver", "审批人", "", "报销审批人"),
        ("t_expense_claim", "description", "费用说明", "", "报销费用说明"),

        # t_fixed_asset
        ("t_fixed_asset", "asset_code", "资产编码", "资产编号", "固定资产唯一编码"),
        ("t_fixed_asset", "asset_name", "资产名称", "设备名称", "固定资产名称"),
        ("t_fixed_asset", "asset_category", "资产类别", "", "房屋建筑/机器设备/运输工具/电子设备/办公家具/其他"),
        ("t_fixed_asset", "acquisition_date", "购入日期", "", "固定资产购入日期"),
        ("t_fixed_asset", "acquisition_cost", "原值", "购置原值", "固定资产原始购置成本"),
        ("t_fixed_asset", "salvage_value", "残值", "预计残值", "固定资产预计净残值"),
        ("t_fixed_asset", "useful_life_months", "使用月数", "折旧月数", "预计使用月数"),
        ("t_fixed_asset", "monthly_depreciation", "月折旧额", "", "每月应计提折旧额"),
        ("t_fixed_asset", "accumulated_depreciation", "累计折旧", "", "截至当前的累计折旧额"),
        ("t_fixed_asset", "depreciation_method", "折旧方法", "", "直线法/双倍余额递减法/年数总和法"),
        ("t_fixed_asset", "location", "存放地点", "使用部门", "资产存放位置"),
        ("t_fixed_asset", "custodian", "保管人", "", "资产保管责任人"),
        ("t_fixed_asset", "status", "资产状态", "", "在用/闲置/已报废/已处置"),
    ]


def seed_data(conn):
    """Seed semantic model for financial and management tables.

    Column names match seed_financial.py and the active database schemas.
    """
    records = semantic_model_records()

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
    print(f"Seeded {len(records)} semantic model business entries")


def clear_table_metadata_cache():
    """Clear schema caches so select_tables sees updated comments and FKs."""
    try:
        import redis as redis_mod
        addr = settings.redis.addr
        host, port = addr.rsplit(":", 1) if ":" in addr else (addr, "6379")
        client = redis_mod.Redis(
            host=host,
            port=int(port),
            db=settings.redis.db,
            password=settings.redis.password or None,
            decode_responses=True,
            socket_timeout=2,
        )
        deleted = client.delete("schema:table_metadata")
        semantic_keys = list(client.scan_iter("schema:semantic_model:*"))
        if semantic_keys:
            deleted += client.delete(*semantic_keys)
        print(f"Cleared {deleted} Redis schema cache keys")
    except Exception as e:
        print(f"Skipped Redis table metadata cache clear: {e}")


def seed_table_visible_semantics(conn):
    """Backfill table comments used by select_tables.

    Existing databases may have been created before table comments were added.
    Missing tables are ignored so the seed remains safe across environments.
    """
    updated = 0
    with conn.cursor() as cur:
        for table_name, comment in table_visible_semantics().items():
            try:
                cur.execute(f"ALTER TABLE `{table_name}` COMMENT=%s", (comment,))
                updated += 1
            except Exception as e:
                print(f"  [SKIP] table comment {table_name}: {e}")
    conn.commit()
    clear_table_metadata_cache()
    print(f"Updated {updated} table visible semantics")


def seed_logical_foreign_keys(conn):
    """更新逻辑外键关系（数据库未定义 FK 但业务上存在的关联）。

    格式：(table_name, column_name, ref_table, ref_column)
    """
    logical_fks = [
        # t_journal_item
        ("t_journal_item", "entry_id", "t_journal_entry", "id"),
        ("t_journal_item", "account_code", "t_account", "account_code"),
        ("t_journal_item", "cost_center_id", "t_cost_center", "id"),

        # t_budget
        ("t_budget", "cost_center_id", "t_cost_center", "id"),
        ("t_budget", "account_code", "t_account", "account_code"),

        # t_expense_claim
        ("t_expense_claim", "cost_center_id", "t_cost_center", "id"),
        ("t_expense_claim", "department_id", "t_department", "id"),

        # organization / access management
        ("t_cost_center", "department_id", "t_department", "id"),
        ("t_department", "parent_id", "t_department", "id"),
        ("t_user_role", "user_id", "t_user", "id"),
        ("t_user_role", "role_id", "t_role", "id"),
        ("t_user_department", "user_id", "t_user", "id"),
        ("t_user_department", "department_id", "t_department", "id"),

        # t_receivable_payable / t_invoice
        ("t_receivable_payable", "related_invoice_id", "t_invoice", "id"),
        ("t_invoice", "related_entry_id", "t_journal_entry", "id"),
    ]

    with conn.cursor() as cur:
        for table_name, column_name, ref_table, ref_column in logical_fks:
            cur.execute(
                """UPDATE t_semantic_model
                   SET is_fk = 1, ref_table = %s, ref_column = %s
                   WHERE table_name = %s AND column_name = %s""",
                (ref_table, ref_column, table_name, column_name),
            )
    conn.commit()
    print(f"Updated {len(logical_fks)} logical foreign keys")


def main():
    conn = get_conn()
    try:
        create_table(conn)
        seed_table_visible_semantics(conn)
        sync_schema_from_information_schema(conn)
        seed_data(conn)
        seed_logical_foreign_keys(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
