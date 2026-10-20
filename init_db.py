"""从项目自带的 MySQL 地区数据转存 SQLite，不需要 MySQL 服务。"""
import os
import re
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import config


SCHEMA = """
CREATE TABLE region (
    id INTEGER PRIMARY KEY,
    name TEXT,
    statis_code TEXT,
    code TEXT,
    full_name TEXT,
    region_type INTEGER,
    parent_id INTEGER
);
CREATE INDEX idx_region_parent ON region(region_type, parent_id);
CREATE INDEX idx_region_name ON region(region_type, name);
CREATE INDEX idx_region_code ON region(code);
"""


def region_inserts(sql_path):
    """仅提取原始文件中 region 表的 INSERT，跳过 MySQL 专用建表和设置。"""
    statement = []
    with Path(sql_path).open(encoding="utf-8") as source:
        for line in source:
            if not statement:
                if not re.match(r"\s*insert\s+into\b", line, re.IGNORECASE):
                    continue
            statement.append(line)
            if line.rstrip().endswith(";"):
                sql = "".join(statement)
                if not sqlite3.complete_statement(sql):
                    continue
                if not re.match(r"\s*insert\s+into\s+`?region`?\s+values\b", sql, re.IGNORECASE):
                    raise ValueError("地区数据包含不支持的 INSERT 语句")
                yield sql
                statement = []
    if statement:
        raise ValueError("地区 SQL 中存在未结束的 INSERT 语句")


def initialize_database(sql_path=None, db_path=None):
    """首次导入；已有非空地区库直接复用，不覆盖本地修改。返回地区数量。"""
    source = Path(sql_path) if sql_path is not None else config.REGION_SQL_PATH
    target = Path(db_path) if db_path is not None else config.SQLITE_DB_PATH
    if target.exists():
        with closing(sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            count = connection.execute("SELECT COUNT(*) FROM region").fetchone()[0]
            if not count:
                raise ValueError(f"地区数据库为空：{target}")
            return count

    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_path = tempfile.mkstemp(prefix="region-", suffix=".db", dir=target.parent)
    os.close(handle)
    try:
        with closing(sqlite3.connect(temporary_path)) as connection:
            connection.executescript(SCHEMA)
            with connection:
                for statement in region_inserts(source):
                    connection.execute(statement)
                count = connection.execute("SELECT COUNT(*) FROM region").fetchone()[0]
                if not count:
                    raise ValueError("地区 SQL 未包含有效数据")
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("SQLite 数据库完整性检查失败")
        os.replace(temporary_path, target)
        return count
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


if __name__ == "__main__":
    count = initialize_database()
    print(f"SQLite 地区数据库已就绪：{config.SQLITE_DB_PATH}，共 {count} 条地区记录")
