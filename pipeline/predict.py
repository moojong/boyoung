"""학습한 모델로 예측하고 결과를 확인한다.

    python predict.py              전체 로트의 예측과 실제를 나란히 본다
    python predict.py 26 27 28     벤토나이트 값별 판정을 본다

학습 설정은 classify.py 를 그대로 쓴다. 이 파일은 조회만 한다.
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.pipeline import make_pipeline

import classify as C
import preprocess as P

FEATURES = ["bentonite"]
MODEL = "xgboost"
WEIGHT = "balanced"


def fit(data: pd.DataFrame | None = None):
    """가진 데이터 전부로 학습한다.

    반환: (모델, 판정 임계값, 판정 기준 불량률, 학습에 쓴 데이터)

    임계값은 학습 구간의 양성 비율에서 산출한다 — 검증을 보지 않는다.
    """
    if data is None:
        data, _ = P.build()
    data = data.sort_values("target_date").reset_index(drop=True)

    rate_cut = C.threshold(data["bad_rate"])
    y = C.label(data["bad_rate"], rate_cut)
    X = data[FEATURES]

    prob_cut = C.cut_rule(X, y, weight=WEIGHT, model=MODEL)
    model = make_pipeline(C.impute(FEATURES), C.candidates(0, WEIGHT)[MODEL])
    model.fit(X, y)
    return model, prob_cut, rate_cut, data


def by_value(values=None) -> pd.DataFrame:
    """벤토나이트 값별 판정. 값 하나가 곧 하나의 답이 된다.

    변수가 하나뿐이라 같은 값을 가진 로트는 모두 같은 판정을 받는다.
    """
    model, prob_cut, rate_cut, data = fit()
    if values is None:
        values = sorted(data["bentonite"].dropna().unique())

    X = pd.DataFrame({"bentonite": [float(v) for v in values]})
    prob = model.predict_proba(X)[:, 1]

    seen = data["bentonite"].dropna().round(0)
    out = pd.DataFrame({
        "벤토나이트": [int(v) for v in values],
        "위험확률": prob.round(3),
        "판정": np.where(prob >= prob_cut, "불량 예상", "정상"),
        "학습 로트": [int((seen == round(float(v))).sum()) for v in values],
    })
    out["학습 범위"] = np.where(out["학습 로트"] > 0, "안", "**밖 — 판정 보류**")
    out.attrs["prob_cut"] = prob_cut
    out.attrs["rate_cut"] = rate_cut
    return out


def review(data: pd.DataFrame | None = None) -> pd.DataFrame:
    """학습에 쓴 로트를 되짚어 본다.

    학습 데이터를 다시 맞히는 것이므로 **성능 평가가 아니다.**
    어떤 로트에서 어긋나는지 눈으로 보기 위한 표다.
    실제 성능은 classify.timed() 의 시간순 검증으로 본다.
    """
    model, prob_cut, rate_cut, data = fit(data)
    y = C.label(data["bad_rate"], rate_cut).values
    prob = model.predict_proba(data[FEATURES])[:, 1]
    pred = (prob >= prob_cut).astype(int)

    out = pd.DataFrame({
        "로트": data["lot_key"],
        "일자": pd.to_datetime(data["target_date"]).dt.strftime("%m-%d"),
        "벤토나이트": data["bentonite"],
        "위험확률": prob.round(3),
        "예측": np.where(pred == 1, "불량", "정상"),
        "실제 불량률": data["bad_rate"].round(2),
        "실제": np.where(y == 1, "불량", "정상"),
    })
    out["결과"] = np.select(
        [(pred == 1) & (y == 1), (pred == 0) & (y == 0), (pred == 1) & (y == 0)],
        ["맞음", "맞음", "헛경보"], default="놓침")
    out.attrs["prob_cut"] = prob_cut
    out.attrs["rate_cut"] = rate_cut
    return out


def _main(argv: list[str]) -> None:
    if argv:
        t = by_value([float(a) for a in argv])
        print(f"판정 기준  불량률 > {t.attrs['rate_cut']:.2f}%   "
              f"위험확률 임계값 {t.attrs['prob_cut']:.3f}\n")
        print(t.to_string(index=False))
        return

    t = by_value()
    print(f"판정 기준  불량률 > {t.attrs['rate_cut']:.2f}%   "
          f"위험확률 임계값 {t.attrs['prob_cut']:.3f}\n")
    print("── 벤토나이트 값별 판정 ──")
    print(t.to_string(index=False))

    r = review()
    print(f"\n── 학습 로트 {len(r)}개 되짚기 (성능 평가 아님) ──")
    print(r["결과"].value_counts().to_frame("로트").to_string())
    miss = r[r["결과"] != "맞음"]
    if len(miss):
        print(f"\n어긋난 {len(miss)}개")
        print(miss.drop(columns="결과").to_string(index=False))
    print("\n실제 성능은 시간순 검증으로 본다 — classify.timed()")


if __name__ == "__main__":
    _main(sys.argv[1:])
