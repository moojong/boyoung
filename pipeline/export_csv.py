"""DB 테이블을 CSV 로 내보낸다.

    python export_csv.py [저장 경로]

엑셀에서 한글이 깨지지 않도록 BOM 을 붙인다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import db

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/Project/boyoung/document/data")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = db.query("SHOW TABLES").iloc[:, 0].tolist()

    for name in tables:
        frame = db.table(name)
        path = OUT / f"{name}.csv"
        # utf-8-sig: 엑셀이 인코딩을 잘못 잡아 한글이 깨지는 것을 막는다
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  {name:22} {len(frame):5}행 × {frame.shape[1]:2}컬럼   "
              f"{path.stat().st_size / 1024:6.0f}KB")


if __name__ == "__main__":
    main()
