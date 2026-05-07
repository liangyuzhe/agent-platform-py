"""Schema Sync：t_semantic_model 全量初始化 + binlog 增量同步 + Redis 缓存。

启动时：
1. 检查 t_semantic_model 是否存在，不存在则建表
2. 检查是否有数据，无数据则全量同步
3. 全量同步后写入 Redis 缓存
4. 启动后台任务监听 binlog DDL 事件，增量更新

运行时：
- binlog 监听 CREATE TABLE / ALTER TABLE / DROP TABLE
- 定时轮询 information_schema 作为 fallback（binlog 不可用时）
- 增量同步后更新 Redis 缓存
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from agents.config.settings import settings

logger = logging.getLogger(__name__)

# DDL 关键词，用于从 binlog query 中识别 DDL 操作
_DDL_KEYWORDS = ("CREATE TABLE", "ALTER TABLE", "DROP TABLE", "RENAME TABLE")

# 轮询间隔（秒），binlog 不可用时的 fallback
_POLL_INTERVAL = 300  # 5 分钟

# Redis key（与 retriever.py 一致）
_REDIS_KEY_TABLE_META = "schema:table_metadata"
_REDIS_KEY_SEMANTIC_PREFIX = "schema:semantic_model:"


def _get_sync_redis():
    """获取同步 Redis 客户端（单例）。失败返回 None。"""
    try:
        from agents.rag.retriever import _get_sync_redis as _get
        return _get()
    except Exception:
        return None


def _refresh_redis_cache(table_names: list[str] | None = None) -> None:
    """同步后刷新 Redis 缓存。

    table_names=None: 全量刷新 table_metadata + 所有表的 semantic_model。
    table_names=[...]: 只刷新指定表 + 更新 table_metadata。
    """
    r = _get_sync_redis()
    if not r:
        return

    try:
        # 刷新 table_metadata
        from agents.rag.retriever import _load_table_metadata_from_mysql
        meta = _load_table_metadata_from_mysql()
        if meta:
            r.set(_REDIS_KEY_TABLE_META, json.dumps(meta, ensure_ascii=False))
            logger.info("Redis: refreshed table_metadata (%d tables)", len(meta))

        # 刷新 semantic_model
        if table_names is None:
            # 全量：获取所有表名
            conn = _get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT table_name FROM t_semantic_model")
                    table_names = [_lower_keys(row)["table_name"] for row in cur.fetchall()]
            finally:
                conn.close()

        if table_names:
            from agents.rag.retriever import _load_semantic_model_from_mysql
            semantic = _load_semantic_model_from_mysql(table_names)
            if semantic:
                pipe = r.pipeline()
                for t, cols in semantic.items():
                    pipe.set(f"{_REDIS_KEY_SEMANTIC_PREFIX}{t}", json.dumps(cols, ensure_ascii=False))
                pipe.execute()
                logger.info("Redis: refreshed semantic_model for %d tables", len(semantic))
    except Exception as e:
        logger.warning("Redis cache refresh failed: %s", e)


def _get_conn() -> pymysql.Connection:
    return pymysql.connect(
        host=settings.mysql.host,
        port=settings.mysql.port,
        user=settings.mysql.username,
        password=settings.mysql.password,
        database=settings.mysql.database,
        charset="utf8mb4",
        cursorclass=DictCursor,
    )


def _lower_keys(row: dict) -> dict:
    """将字典的 key 转为小写（MySQL DictCursor 返回大写 key）。"""
    return {k.lower(): v for k, v in row.items()}


def ensure_semantic_model_table(conn: pymysql.Connection | None = None) -> None:
    """确保 t_semantic_model 表存在，不存在则创建。"""
    close = conn is None
    if conn is None:
        conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS t_semantic_model (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    table_name VARCHAR(128) NOT NULL COMMENT '物理表名',
                    column_name VARCHAR(128) NOT NULL COMMENT '物理字段名',
                    column_type VARCHAR(128) COMMENT '字段类型',
                    column_comment VARCHAR(512) COMMENT '字段注释',
                    is_pk TINYINT DEFAULT 0 COMMENT '是否主键',
                    is_fk TINYINT DEFAULT 0 COMMENT '是否外键',
                    ref_table VARCHAR(128) COMMENT '外键引用表',
                    ref_column VARCHAR(128) COMMENT '外键引用字段',
                    business_name VARCHAR(256) COMMENT '业务名称',
                    synonyms TEXT COMMENT '同义词，逗号分隔',
                    business_description TEXT COMMENT '业务描述',
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
                    pass
        conn.commit()
        logger.info("t_semantic_model table ensured")
    finally:
        if close:
            conn.close()


def _sync_single_table(conn: pymysql.Connection, table_name: str) -> int:
    """同步单张表的 schema 到 t_semantic_model。

    Returns: 更新的字段数。
    """
    db = settings.mysql.database

    # 获取字段信息
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, column_type, column_comment, column_key "
            "FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s "
            "ORDER BY ordinal_position",
            (db, table_name),
        )
        columns = cur.fetchall()

    if not columns:
        # 表不存在或无字段，删除该表的记录
        with conn.cursor() as cur:
            cur.execute("DELETE FROM t_semantic_model WHERE table_name = %s", (table_name,))
        conn.commit()
        logger.info("Removed t_semantic_model entries for dropped table: %s", table_name)
        return 0

    # 获取外键
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, referenced_table_name, referenced_column_name "
            "FROM information_schema.key_column_usage "
            "WHERE table_schema = %s AND table_name = %s AND referenced_table_name IS NOT NULL",
            (db, table_name),
        )
        fk_rows = [_lower_keys(r) for r in cur.fetchall()]
    fk_map = {r["column_name"]: (r["referenced_table_name"], r["referenced_column_name"]) for r in fk_rows}

    # 获取当前表在 t_semantic_model 中的列（用于检测删除的列）
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM t_semantic_model WHERE table_name = %s",
            (table_name,),
        )
        existing_cols = {_lower_keys(r)["column_name"] for r in cur.fetchall()}

    current_cols = set()
    updated = 0
    with conn.cursor() as cur:
        for col in columns:
            col = _lower_keys(col)
            col_name = col["column_name"]
            current_cols.add(col_name)
            is_pk = 1 if col.get("column_key") == "PRI" else 0
            ref_tbl, ref_col = fk_map.get(col_name, (None, None))
            is_fk = 1 if ref_tbl else 0

            cur.execute(
                """INSERT INTO t_semantic_model
                   (table_name, column_name, column_type, column_comment, is_pk, is_fk, ref_table, ref_column)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                       column_type=VALUES(column_type),
                       column_comment=VALUES(column_comment),
                       is_pk=VALUES(is_pk),
                       is_fk=VALUES(is_fk),
                       ref_table=VALUES(ref_table),
                       ref_column=VALUES(ref_column)""",
                (table_name, col_name, col.get("column_type"), col.get("column_comment"),
                 is_pk, is_fk, ref_tbl, ref_col),
            )
            updated += 1

    # 删除已不存在的列
    removed_cols = existing_cols - current_cols
    if removed_cols:
        placeholders = ", ".join(["%s"] * len(removed_cols))
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM t_semantic_model WHERE table_name = %s AND column_name IN ({placeholders})",
                [table_name] + list(removed_cols),
            )

    conn.commit()
    return updated


def full_sync() -> dict[str, Any]:
    """全量同步所有表的 schema 到 t_semantic_model。

    Returns: {"tables": int, "columns": int}
    """
    db = settings.mysql.database
    conn = _get_conn()
    try:
        ensure_semantic_model_table(conn)

        # 获取所有用户表
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s AND table_type = 'BASE TABLE'",
                (db,),
            )
            tables = [_lower_keys(r)["table_name"] for r in cur.fetchall()]

        total_columns = 0
        for tbl in tables:
            count = _sync_single_table(conn, tbl)
            total_columns += count

        result = {"tables": len(tables), "columns": total_columns}
        logger.info("Full schema sync done: %d tables, %d columns", result["tables"], result["columns"])
        return result
    finally:
        conn.close()


def sync_tables(table_names: list[str]) -> int:
    """增量同步指定表的 schema。

    Returns: 更新的字段总数。
    """
    if not table_names:
        return 0
    conn = _get_conn()
    try:
        total = 0
        for tbl in table_names:
            total += _sync_single_table(conn, tbl)
        return total
    finally:
        conn.close()


def _is_ddl_query(query: str) -> list[str] | None:
    """检查 binlog query 是否是 DDL 语句。

    Returns: 被影响的表名列表，或 None（非 DDL）。
    """
    upper = query.upper().strip()
    # 跳过内部表
    if "t_semantic_model" in upper:
        return None

    for kw in _DDL_KEYWORDS:
        if kw in upper:
            # 尝试提取表名（简单解析）
            # CREATE TABLE IF NOT EXISTS table_name ...
            # ALTER TABLE table_name ...
            # DROP TABLE IF EXISTS table_name ...
            # RENAME TABLE old TO new, ...
            import re
            if kw == "RENAME TABLE":
                # RENAME TABLE old TO new, old2 TO new2
                matches = re.findall(r"RENAME\s+TABLE\s+(.+)", upper, re.IGNORECASE)
                if matches:
                    parts = matches[0].split(",")
                    tables = []
                    for p in parts:
                        tokens = p.strip().split()
                        if tokens:
                            tables.append(tokens[0].lower())
                    return tables
            else:
                # CREATE/ALTER/DROP TABLE [IF NOT EXISTS/IF EXISTS] table_name
                pattern = kw + r"\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?(\S+)"
                match = re.search(pattern, upper, re.IGNORECASE)
                if match:
                    table = match.group(1).strip("`").strip('"')
                    # 跳过内部表
                    if table.lower() != "t_semantic_model":
                        return [table.lower()]
            return None
    return None


def _binlog_listener_sync(logger: logging.Logger) -> None:
    """同步版 binlog 监听（在线程池中运行）。"""
    try:
        from pymysqlreplication import BinLogStreamReader
        from pymysqlreplication.event import QueryEvent
    except ImportError:
        logger.warning("mysql-replication not installed, binlog listener disabled")
        return

    mysql_settings = {
        "host": settings.mysql.host,
        "port": settings.mysql.port,
        "user": settings.mysql.username,
        "passwd": settings.mysql.password,
    }

    stream = None
    try:
        # 获取当前 binlog 位置（兼容 MySQL 8.0.22+ 和旧版本）
        conn = _get_conn()
        with conn.cursor() as cur:
            try:
                cur.execute("SHOW BINARY LOG STATUS")
            except Exception:
                cur.execute("SHOW MASTER STATUS")
            status = cur.fetchone()
        conn.close()

        if not status:
            logger.warning("Cannot get binlog status, binlog listener disabled")
            return

        log_file = status["File"]
        log_pos = status["Position"]
        logger.info("Starting binlog listener from %s:%s", log_file, log_pos)

        stream = BinLogStreamReader(
            connection_settings=mysql_settings,
            server_id=1000,  # 唯一 server_id
            log_file=log_file,
            log_pos=log_pos,
            only_events=[QueryEvent],
            blocking=True,
            resume_stream=True,
        )

        for binlogevent in stream:
            if isinstance(binlogevent, QueryEvent):
                query = binlogevent.query
                affected_tables = _is_ddl_query(query)
                if affected_tables:
                    logger.info("DDL detected: %s -> tables: %s", query[:100], affected_tables)
                    try:
                        count = sync_tables(affected_tables)
                        logger.info("Incremental sync for %s: %d columns updated", affected_tables, count)
                        _refresh_redis_cache(affected_tables)
                    except Exception as e:
                        logger.warning("Incremental sync failed for %s: %s", affected_tables, e)

    except Exception as e:
        logger.warning("Binlog listener error: %s", e)
    finally:
        if stream:
            stream.close()


async def _polling_fallback(logger: logging.Logger) -> None:
    """定时轮询 information_schema 作为 fallback。"""
    db = settings.mysql.database

    # 记录上次同步的表结构快照（table_name -> 最大 updated_at）
    last_snapshot: dict[str, str] = {}

    # 初始化快照
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name, MAX(updated_at) as max_updated "
                "FROM t_semantic_model GROUP BY table_name"
            )
            for row in cur.fetchall():
                row = _lower_keys(row)
                last_snapshot[row["table_name"]] = str(row["max_updated"])
        conn.close()
    except Exception as e:
        logger.warning("Failed to init polling snapshot: %s", e)

    while True:
        await asyncio.sleep(_POLL_INTERVAL)
        try:
            conn = _get_conn()
            with conn.cursor() as cur:
                # 获取当前所有表
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = %s AND table_type = 'BASE TABLE'",
                    (db,),
                )
                current_tables = {_lower_keys(r)["table_name"] for r in cur.fetchall()}

                # 获取 t_semantic_model 中的表
                cur.execute("SELECT DISTINCT table_name FROM t_semantic_model")
                tracked_tables = {_lower_keys(r)["table_name"] for r in cur.fetchall()}

            conn.close()

            # 检查新增或删除的表
            new_tables = current_tables - tracked_tables
            dropped_tables = tracked_tables - current_tables

            if new_tables:
                logger.info("Polling detected new tables: %s", new_tables)
                sync_tables(list(new_tables))
                _refresh_redis_cache(list(new_tables))

            if dropped_tables:
                logger.info("Polling detected dropped tables: %s", dropped_tables)
                conn = _get_conn()
                try:
                    with conn.cursor() as cur:
                        for tbl in dropped_tables:
                            cur.execute("DELETE FROM t_semantic_model WHERE table_name = %s", (tbl,))
                    conn.commit()
                finally:
                    conn.close()
                _refresh_redis_cache()

            # TODO: 检查列变更（需要对比 column 信息）

        except Exception as e:
            logger.warning("Polling fallback error: %s", e)


async def start_schema_sync(logger: logging.Logger | None = None) -> asyncio.Task:
    """启动 schema 同步后台任务。

    1. 全量同步（如果需要）
    2. 启动 binlog 监听
    3. 启动定时轮询 fallback

    Returns: 后台 Task 对象。
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # 检查是否需要全量同步（在线程池中执行，不阻塞事件循环）
    async def _check_and_sync():
        try:
            conn = _get_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as cnt FROM t_semantic_model")
                count = _lower_keys(cur.fetchone())["cnt"]
            conn.close()
            if count == 0:
                logger.info("t_semantic_model is empty, running full sync...")
                result = await asyncio.to_thread(full_sync)
                logger.info("Full sync result: %s", result)
            else:
                logger.info("t_semantic_model has %d entries, skipping full sync", count)
        except Exception as e:
            logger.warning("Failed to check t_semantic_model, running full sync: %s", e)
            try:
                result = await asyncio.to_thread(full_sync)
                logger.info("Full sync result: %s", result)
            except Exception as e2:
                logger.error("Full sync failed: %s", e2)

        # 无论全量同步与否，都刷新 Redis 缓存（确保 Redis 有数据）
        try:
            await asyncio.to_thread(_refresh_redis_cache)
        except Exception as e:
            logger.warning("Redis cache refresh failed: %s", e)

    # 启动后台任务
    async def _background():
        # 先执行全量同步检查
        await _check_and_sync()
        # 并行运行 binlog 监听和轮询 fallback（binlog 是阻塞的，放线程池）
        await asyncio.gather(
            asyncio.to_thread(_binlog_listener_sync, logger),
            _polling_fallback(logger),
            return_exceptions=True,
        )

    task = asyncio.create_task(_background())
    logger.info("Schema sync background task started")
    return task
