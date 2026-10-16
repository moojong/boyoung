"""주물사 시험 데이터와 불량 집계를 로트 단위로 이어 학습 데이터를 만든다.

두 시스템의 로트번호 표기가 달라 그대로는 붙지 않는다.
  시험결과   6AB-1, 6AB-2, 6BC 1   같은 로트를 하루에 여러 번 재고 차수를 붙였다
  불량집계   6AB, 6BC              차수 없이 로트만 기록한다
차수를 떼야 짝이 맞는다. 떼지 않으면 95개 중 1개만 붙는다.

로트번호는 제품을 넘어 재사용되므로(같은 6AB 가 9일 뒤 다른 제품에 쓰인다)
제품을 먼저 거르지 않으면 남의 제품 시험값이 딸려온다.
"""

from __future__ import annotations

import pandas as pd

import db

PRODUCT = "4G100 CUP (CV5)"

# 같은 제품이 시기·시스템마다 다르게 적혀 있다 (2026-09-30 현장 확인).
# 1~2월 시험값은 '4G100', 1~4월 불량 집계는 '4G 100 컵' 으로 들어왔다.
# 4G150 등 다른 제품과 로트번호가 겹치므로 정확한 목록으로만 거른다.
PRODUCT_ALIASES = {"4G100 CUP (CV5)", "4G100", "4G100 CUP #2", "4G 100 컵", "4G 100 컵 #2"}

# (접두, 한글명). 모래온도만 규격이 없다.
# DB 스키마상 bent_val 은 "벤토나이트"(BENT_VALU) 한 컬럼이다.
# '28/1' 처럼 두 값이 들어 있는데, 뒤 숫자(1·2)는 모래온도와 0.785 로 겹쳐
# 계절을 대신 담을 뿐 예측에 기여하지 않는다. 앞 숫자만 쓴다.
BENT = [("bentonite", "벤토나이트")]

MEASURES = [
    ("sand_temp", "모래온도"),
    ("moist", "수분"),
    ("fill", "충진율"),
    ("cstr", "압축강도"),
    ("sstr", "전단강도"),
    ("perm", "통기도"),
    ("fswt", "충진시편"),
    ("sswt", "표준시편"),
]
SPEC = [p for p, _ in MEASURES if p != "sand_temp"]

# 불량 유형. 뒤 넷은 발생이 15로트 이하라 개별 학습이 어렵다.
DEFECTS = {
    "bad_mix": "사혼입",
    "bad_core": "중자삽입불량",
    "bad_fill": "미충진",
    "bad_crack": "몰드크랙",
    "bad_ecc": "편심/팽창",
    "bad_crush": "짓눌림",
    "bad_gas": "가스",
    "bad_jig": "지그불량",
}
MAIN_DEFECTS = ["bad_mix", "bad_core", "bad_fill", "bad_crack"]

# 집계 방식과 그 밖의 컬럼을 한글로 옮기기 위한 표
_HOW = {"mean": "평균", "max": "최대", "min": "최소", "sum": "합", "mode": ""}
_ETC = {
    "lot_key": "로트",
    "n_tests": "시험횟수",
    "night_ratio": "야간비율",
    "production_qty": "생산수량",
    "bad_qty": "불량수량",
    "bad_rate": "불량률",
    "target_date": "집계일자",
    "month": "월",
}


def ko(name: str) -> str:
    """컬럼명을 한글로 옮긴다. 'sand_temp_mean' -> '모래온도 평균'.

    집계 결과 컬럼이 서른 개가 넘어 원문만으로는 읽기 어렵다.
    """
    if name in _ETC:
        return _ETC[name]
    if name in DEFECTS:
        return DEFECTS[name]
    if name.endswith("_rate") and name[:-5] in DEFECTS:
        return f"{DEFECTS[name[:-5]]} 비율"

    labels = dict(MEASURES) | dict(BENT)
    for pre, label in labels.items():
        if not name.startswith(pre):
            continue
        rest = name[len(pre):].lstrip("_")
        if not rest:                                # bentonite — 접미사 없음
            return label
        if rest.startswith("val_"):                 # moist_val_min
            return f"{label} {_HOW.get(rest[4:], rest[4:])}".strip()
        if rest.startswith("out_"):                 # moist_out_sum
            return f"{label} 이탈횟수"
        if rest in _HOW:                            # sand_temp_mean
            return f"{label} {_HOW[rest]}".strip()
    return name


def ko_en(name: str) -> str:
    """한글과 원문을 함께 보여준다. '모래온도 평균 (sand_temp_mean)'.

    한글만 두면 코드와 대조가 안 되고, 원문만 두면 읽기 어렵다.
    """
    label = ko(name)
    return label if label == name else f"{label} ({name})"


def ko_frame(df: pd.DataFrame, axis: str = "both", en: bool = False) -> pd.DataFrame:
    """표의 컬럼명과 인덱스를 한글로 바꿔 보여준다. 원본은 건드리지 않는다."""
    label = ko_en if en else ko
    out = df.copy()
    if axis in ("both", "columns"):
        out.columns = [label(str(c)) for c in out.columns]
    if axis in ("both", "index"):
        out.index = [label(str(i)) for i in out.index]
    return out


def lot_key(s: pd.Series) -> pd.Series:
    """로트번호에서 차수를 떼고 표기를 통일한다. '6AB-1', '6BC 1', '6IT1' -> '6AB', '6BC', '6IT'.

    차수는 생산량이 많아 로트를 나눈 것이다 (2026-09-30 현장 확인).
    """
    return (
        s.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"[\s\-]+\d+$", "", regex=True)   # 끝에 붙은 차수
        .str.replace(r"[\s\-]+", "", regex=True)       # 남은 구분자
        .str.replace(r"^(\d[A-Z]{2})\d$", r"\1", regex=True)  # 붙여 쓴 차수 6IT1 -> 6IT
        .replace({"": pd.NA})
    )


def _split_bent(df: pd.DataFrame) -> pd.DataFrame:
    """bent_val 에서 앞 숫자를 뽑는다. '28/1' -> 28.

    앞 숫자 = 벤토나이트 투입량, 뒤 숫자 = 시퀀스 (2026-09-30 현장 확인). 뒤 숫자는 버린다.
    """
    out = df.copy()
    parts = out["bent_val"].astype(str).str.extract(
        r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")
    out["bentonite"] = pd.to_numeric(parts[0], errors="coerce")
    return out


def _add_spec_flags(df: pd.DataFrame) -> pd.DataFrame:
    """규격 이탈 여부를 만든다.

    규격이 전 품목 고정이라 '규격 대비 위치'는 원값의 선형변환일 뿐이다(상관 1.000).
    경계에서 꺾이는 이탈 여부만 원값과 다른 정보를 준다.
    """
    out = df.copy()
    for pre in SPEC:
        val, lo, hi = out[f"{pre}_val"], out[f"{pre}_min"], out[f"{pre}_max"]
        out[f"{pre}_out"] = ((val < lo) | (val > hi)).astype(float)
    return out


def _mode(s: pd.Series) -> float:
    """가장 많이 나온 값. 같은 수면 작은 쪽."""
    s = s.dropna()
    return float(s.mode().iloc[0]) if len(s) else float("nan")


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """로트 하나를 한 행으로 만든다.

    로트당 시험이 1~4회이고 값이 흔들리므로(수분 4.5->4.1, 충진율 48->41)
    어느 대푯값을 쓸지는 데이터로 정해야 한다. 세 가지를 남긴다.
    """
    values = ["sand_temp"] + [f"{p}_val" for p in SPEC]
    how: dict[str, object] = {c: ["mean", "max", "min", "first"] for c in values}
    # 벤토나이트는 8종짜리 이산 설정값이고 로트 안에서 거의 변하지 않는다
    # (104로트 중 98개가 한 값). 평균을 내면 21·26.5 처럼 실제로 쓰인 적 없는
    # 값이 생기고, 최대·최소는 최빈값과 사실상 중복이다. 최빈값 하나만 둔다.
    how |= {"bentonite": [_mode, "first"]}
    how |= {f"{p}_out": "sum" for p in SPEC}      # 이탈 횟수
    how["ftr_seq"] = "count"                      # 시험 횟수
    how["juya_gubn"] = "mean"                     # 야간 비율

    df = df.sort_values(["lot_key", "real_date", "test_time"])
    g = df.groupby("lot_key").agg(how)
    g.columns = [f"{c}_{getattr(fn, '__name__', fn).lstrip('_')}" for c, fn in g.columns]
    return g.rename(
        columns={"ftr_seq_count": "n_tests", "juya_gubn_mean": "night_ratio",
                 "bentonite_mode": "bentonite"}      # 집계 방식을 이름에 남기지 않는다
    ).reset_index()


def build(product: str = PRODUCT) -> tuple[pd.DataFrame, dict]:
    """학습 데이터와 구성 보고를 돌려준다."""
    foundry = db.table("foundry_test_result")
    bad = db.table("bad_statistics")

    names = PRODUCT_ALIASES if product == PRODUCT else {product}
    f = foundry[foundry["product_name"].isin(names)].copy()
    f["lot_key"] = lot_key(f["lot_no"])
    f["real_date"] = pd.to_datetime(f["real_date"], errors="coerce")
    f["test_time"] = f["test_time"].astype(str)
    f = f.dropna(subset=["lot_key"])
    # 측정을 안 한 칸이 0 으로 들어온다 (6A3 1/28). 0 은 실제 값이 아니다.
    measured = ["sand_temp"] + [f"{p}_val" for p in SPEC]
    f[measured] = f[measured].apply(pd.to_numeric, errors="coerce").replace(0, float("nan"))

    bad = bad[bad["product_name"].isin(names)].copy()
    bad["lot_key"] = lot_key(bad["lot_no"])
    bad["target_date"] = pd.to_datetime(bad["target_date"], errors="coerce")

    features = _aggregate(_add_spec_flags(_split_bent(f)))

    # 한 로트가 여러 날 집계되기도 하므로 로트 단위로 합친다
    targets = bad.groupby("lot_key").agg(
        production_qty=("production_qty", "sum"),
        bad_qty=("bad_qty", "sum"),
        target_date=("target_date", "min"),
        **{t: (t, "sum") for t in DEFECTS},
    ).reset_index()

    # 생산량이 로트마다 크게 달라 개수를 그대로 쓰면 생산량을 학습한다. 비율로 바꾼다.
    targets["bad_rate"] = 100 * targets["bad_qty"] / targets["production_qty"]
    for t in DEFECTS:
        targets[f"{t}_rate"] = 100 * targets[t] / targets["production_qty"]

    data = features.merge(targets, on="lot_key").sort_values("target_date")
    data["month"] = data["target_date"].dt.month

    report = {
        "시험 로트": len(features),
        "불량 로트": len(targets),
        "매칭": len(data),
        "매칭률": f"{len(data) / len(targets):.0%}",
        "기간": f"{data['target_date'].min():%Y-%m-%d} ~ {data['target_date'].max():%Y-%m-%d}",
    }
    return data.reset_index(drop=True), report


def feature_columns(data: pd.DataFrame) -> list[str]:
    """학습에 쓸 수 있는 변수만 고른다. 타겟과 값이 하나뿐인 것은 뺀다."""
    exclude = {"production_qty", "bad_qty", "bad_rate", "month"}
    exclude |= set(DEFECTS) | {f"{t}_rate" for t in DEFECTS}
    cols = [c for c in data.select_dtypes("number").columns if c not in exclude]
    return [c for c in cols if data[c].nunique() > 1]


if __name__ == "__main__":
    data, rep = build()
    for k, v in rep.items():
        print(f"  {k:8} {v}")
    print(f"  {'변수':8} {len(feature_columns(data))}개")
