"""보영 MySQL 조회 헬퍼.

pymysql 설치가 시스템 정책(PEP 668)으로 막혀 있어 docker mysql 클라이언트를 거친다.
읽기 전용으로만 쓴다.

접속 정보는 환경변수로 받는다. 없으면 저장소 루트의 `.env` 에서 읽는다 (git 에 올리지 않는다).
  BYHT_HOST BYHT_PORT BYHT_USER BYHT_PASSWORD BYHT_DB    — 형식은 `.env.example`
"""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import pandas as pd


def _load_env(path: Path = Path(__file__).resolve().parent.parent / ".env") -> None:
    """`.env` 의 KEY=VALUE 를 환경변수로 올린다. 이미 설정된 값은 덮어쓰지 않는다."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

HOST = os.getenv("BYHT_HOST", "")
PORT = os.getenv("BYHT_PORT", "")
USER = os.getenv("BYHT_USER", "")
PASSWORD = os.getenv("BYHT_PASSWORD", "")
DATABASE = os.getenv("BYHT_DB", "byht_db")

IMAGE = "mysql:8"
TIMEOUT = 120


def query(sql: str) -> pd.DataFrame:
    """SELECT 결과를 DataFrame 으로 돌려준다."""
    if not (HOST and USER and PASSWORD):
        raise RuntimeError("DB 접속 정보가 없다. .env.example 을 복사해 저장소 루트에 .env 를 만든다.")
    proc = subprocess.run(
        ["docker", "run", "--rm", "-e", f"MYSQL_PWD={PASSWORD}", IMAGE,
         "mysql", "-h", HOST, "-P", PORT, "-u", USER, DATABASE,
         "--default-character-set=utf8mb4", "--batch", "--raw", "-e", sql],
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    if proc.returncode != 0:
        # 경고는 stderr 로 항상 나오므로 실패했을 때만 본다
        raise RuntimeError(proc.stderr.strip() or "조회 실패")
    if not proc.stdout.strip():
        return pd.DataFrame()

    return pd.read_csv(
        io.StringIO(proc.stdout),
        sep="\t",
        na_values=["\\N"],
        keep_default_na=True,
    )


def table(name: str, where: str | None = None) -> pd.DataFrame:
    """테이블 전체(또는 조건에 맞는 행)를 가져온다."""
    sql = f"SELECT * FROM {name}"
    if where:
        sql += f" WHERE {where}"
    return query(sql)


def freshness() -> pd.DataFrame:
    """두 테이블이 어디까지 채워져 있는지 확인한다. 동기화 상태 점검용."""
    return query("""
        SELECT 'foundry_test_result' AS 테이블, COUNT(*) AS 행,
               MIN(real_date) AS 시작, MAX(real_date) AS 최종,
               COUNT(DISTINCT lot_no) AS 로트
        FROM foundry_test_result
        UNION ALL
        SELECT 'bad_statistics', COUNT(*), MIN(target_date), MAX(target_date),
               COUNT(DISTINCT lot_no)
        FROM bad_statistics
    """)


if __name__ == "__main__":
    print(freshness().to_string(index=False))
