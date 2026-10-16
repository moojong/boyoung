"""생산·양품·불량 수량 집계.

`bad_statistics` 한 테이블만 쓴다. 공정 변수와 잇지 않고 실적 자체를 본다.

금액·중량 컬럼은 수량에 단가·단중을 곱한 값이라 새 정보가 없다.
보고용 환산에만 쓰고 분석 변수로는 쓰지 않는다.
"""

from __future__ import annotations

import pandas as pd

import db

DEFECTS = {
    "bad_mix": "사혼입",
    "bad_core": "중자삽입불량",
    "bad_fill": "미충진",
    "bad_crack": "몰드크랙",
    "bad_ecc": "편심/팽창",
    "bad_crush": "짓눌림",
    "bad_gas": "가스",
    "bad_jig": "지그불량",
    "bad_mold": "형빼기",
}

# 생산이 사실상 멈춘 달은 비율이 튀어 추세를 가린다
MIN_MONTHLY_QTY = 1_000


def load() -> pd.DataFrame:
    """실적 데이터를 읽고 양품 수량을 더한다."""
    df = db.table("bad_statistics")
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["good_qty"] = df["production_qty"] - df["bad_qty"]
    df["month"] = df["target_date"].dt.to_period("M")

    # 유형 합계와 총 불량의 차이. 어느 유형에도 잡히지 않는 몫이다.
    df["typed_qty"] = df[list(DEFECTS)].sum(axis=1)
    df["untyped_qty"] = df["bad_qty"] - df["typed_qty"]
    return df


def overall(df: pd.DataFrame) -> pd.Series:
    """전체 요약. 불량률은 로트별 평균이 아니라 총량 기준으로 낸다.

    소량 로트를 똑같이 취급하면 비율이 왜곡된다.
    """
    prod, bad = df["production_qty"].sum(), df["bad_qty"].sum()
    return pd.Series({
        "로트": len(df),
        "생산수량": prod,
        "양품수량": df["good_qty"].sum(),
        "불량수량": bad,
        "불량률(%)": round(100 * bad / prod, 2),
        "불량금액": df["bad_amount"].sum(),
        "불량중량": round(df["bad_weight"].sum(), 1),
        "미분류 불량": df["untyped_qty"].sum(),
        "미분류 비중(%)": round(100 * df["untyped_qty"].sum() / bad, 1),
    })


def by_lot(df: pd.DataFrame) -> pd.DataFrame:
    """로트 단위 분포. 평균과 중앙값이 갈리는 방향에서 쏠림이 보인다."""
    cols = ["production_qty", "good_qty", "bad_qty", "bad_rate", "bad_amount"]
    stat = df[cols].describe().T[["mean", "50%", "std", "min", "max"]]
    stat.columns = ["평균", "중앙값", "표준편차", "최소", "최대"]
    stat.index = ["생산수량", "양품수량", "불량수량", "불량률(%)", "불량금액"]
    return stat.round(1)


def by_month(df: pd.DataFrame, min_qty: int = MIN_MONTHLY_QTY) -> pd.DataFrame:
    """월별 실적. 생산이 거의 없는 달은 빼고 본다."""
    g = df.groupby("month").agg(
        로트=("bad_qty", "size"),
        생산=("production_qty", "sum"),
        양품=("good_qty", "sum"),
        불량=("bad_qty", "sum"),
        불량금액=("bad_amount", "sum"),
    )
    g["불량률(%)"] = (100 * g["불량"] / g["생산"]).round(2)
    return g[g["생산"] >= min_qty]


def by_type(df: pd.DataFrame) -> pd.DataFrame:
    """유형별 구성. 미분류 몫을 한 줄로 함께 보여준다."""
    total = df["bad_qty"].sum()
    rows = [{"유형": name, "수량": int(df[col].sum()),
             "발생 로트": int((df[col] > 0).sum())}
            for col, name in DEFECTS.items()]
    rows.append({"유형": "미분류", "수량": int(df["untyped_qty"].sum()),
                 "발생 로트": int((df["untyped_qty"] > 0).sum())})

    out = pd.DataFrame(rows).sort_values("수량", ascending=False).reset_index(drop=True)
    out["전체 대비(%)"] = (100 * out["수량"] / total).round(1)
    out["금액"] = out["수량"] * df["unit_price"].iloc[0]
    return out


def report(df: pd.DataFrame | None = None) -> None:
    """한 번에 훑어보는 요약."""
    df = load() if df is None else df
    o = overall(df)

    print("■ 전체")
    print(f"    기간      {df['target_date'].min():%Y-%m-%d} ~ {df['target_date'].max():%Y-%m-%d}")
    print(f"    로트      {o['로트']:,}개")
    print(f"    생산      {o['생산수량']:>10,}개")
    print(f"    양품      {o['양품수량']:>10,}개   {100 - o['불량률(%)']:.2f}%")
    print(f"    불량      {o['불량수량']:>10,}개   {o['불량률(%)']:.2f}%")
    print(f"    불량금액   {o['불량금액']:>10,.0f}원")

    print("\n■ 로트 단위 분포")
    print(by_lot(df).to_string())

    print("\n■ 월별")
    print(by_month(df).to_string())

    print("\n■ 유형별")
    print(by_type(df).to_string(index=False))
    print(f"\n    미분류 {o['미분류 불량']:,}개 ({o['미분류 비중(%)']}%) — "
          "총 불량과 유형 합계가 맞지 않는다")


if __name__ == "__main__":
    report()
