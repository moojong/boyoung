"""양불 판정 분류 — 후보 모델과 평가를 한 곳에 모은다.

    노트북·스트림릿·스크립트가 모두 이 모듈을 불러 쓴다.
    수치가 세 곳에서 따로 나오는 일을 막기 위한 것이다.

후보 목록은 BADA 의 `ENSEMBLE_MODEL_MAP[TaskType.CLASSIFICATION]` 7종을 따르고
logistic regression 과 기준선을 더했다. 하이퍼파라미터·검증·임계값은 BADA 와 다르다 —
사유는 `candidates` 와 `timed` 주석에 적었다.
"""

from __future__ import annotations

import os

# 표본이 100 미만이라 병렬화 이득이 없고 스레드 경합만 생긴다
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import lightgbm as lgb
import xgboost as xgb
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

BASELINE = "기준선"

THRESHOLDS = {
    "평균": lambda s: s.mean(),
    "중앙값": lambda s: s.median(),
    "상위25%": lambda s: s.quantile(0.75),
}


def threshold(rate: pd.Series, how: str = "평균") -> float:
    """판정 기준값. 기본은 평균 불량률."""
    return float(THRESHOLDS[how](rate))


def label(rate: pd.Series, th: float) -> pd.Series:
    """기준을 넘으면 1(불량 초과), 아니면 0(정상)."""
    return (rate > th).astype(int)


def candidates(seed: int = 0, weight: str | None = None) -> dict:
    """BADA 7종 + logistic regression + 기준선.

    표본이 작아 깊이를 얕게 묶는다. BADA 는 sklearn 기본값을 쓰는데
    `max_depth` 가 None·10·15 면 93행을 전부 암기한다.
    """
    w = weight  # 'balanced' 면 양성에 가중치를 준다
    return {
        BASELINE: DummyClassifier(strategy="most_frequent"),
        "logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight=w, random_state=seed)),
        "linear svm": make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.5, class_weight=w, max_iter=5000, random_state=seed)),
        "decision tree": DecisionTreeClassifier(
            max_depth=2, min_samples_leaf=8, class_weight=w, random_state=seed),
        "random forest": RandomForestClassifier(
            n_estimators=300, max_depth=3, min_samples_leaf=5,
            class_weight=w, random_state=seed, n_jobs=1),
        "gradient boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.05, random_state=seed),
        "adaboost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.05, random_state=seed),
        "lgbm": lgb.LGBMClassifier(
            n_estimators=100, max_depth=3, num_leaves=7, min_child_samples=8,
            learning_rate=0.05, class_weight=w, random_state=seed,
            n_jobs=1, verbose=-1),
        "xgboost": xgb.XGBClassifier(
            n_estimators=100, max_depth=2, min_child_weight=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=seed,
            n_jobs=1, eval_metric="logloss"),
    }


def impute(columns) -> ColumnTransformer:
    """결측 대체 규칙. 컬럼 성격에 따라 다르게 채운다.

    규격 이탈 횟수는 0 으로 채운다. 측정이 없으면 이탈을 **관측하지 못한** 것이지
    이탈한 것이 아니다. 중앙값을 쓰면 fswt_out_sum 이 1 로 채워져 없던 위반이 생긴다.

    나머지는 중앙값. 이상치를 지우지 않고 두므로 평균보다 안전하다.
    `add_indicator` 가 "비어 있었다"는 사실을 변수로 남긴다 —
    결측이 없으면 아무 컬럼도 늘지 않으므로 지금 결과는 바뀌지 않는다.
    """
    columns = list(columns)
    zero = [c for c in columns if c.endswith("_out_sum")]
    median = [c for c in columns if c not in zero]
    steps = []
    if median:
        steps.append(("중앙값", SimpleImputer(strategy="median",
                                           add_indicator=True), median))
    if zero:
        steps.append(("영", SimpleImputer(strategy="constant", fill_value=0,
                                         add_indicator=True), zero))
    return ColumnTransformer(steps)


def _frame(X) -> pd.DataFrame:
    """대체 규칙이 컬럼 이름을 보므로 이름을 잃지 않게 한다."""
    if isinstance(X, pd.DataFrame):
        return X.astype(float)
    X = np.asarray(X, dtype=float)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def _proba(model, X, y, cv):
    """확률을 못 내는 모델은 결정함수로 대신한다. AUC 는 순위만 보므로 문제없다."""
    try:
        return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    except Exception:
        pass
    try:
        return cross_val_predict(model, X, y, cv=cv, method="decision_function")
    except Exception:
        return None


def evaluate(X, y, folds: int = 5, seed: int = 0,
             weight: str | None = None) -> tuple[pd.DataFrame, dict]:
    """교차검증으로 후보를 한 번에 비교한다.

    반환: (성적표, 모델별 예측)
    """
    X = _frame(X)
    y = np.asarray(y, dtype=int)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    fill = impute(X.columns)

    rows, preds = [], {}
    for name, raw in candidates(seed, weight).items():
        # 대체는 폴드 안에서 계산된다. 전체 중앙값을 쓰면 검증 구간이 새어든다.
        model = make_pipeline(clone(fill), raw)
        pred = cross_val_predict(model, X, y, cv=cv)
        prob = _proba(model, X, y, cv)
        preds[name] = pred
        rows.append({
            "모델": name,
            "정확도": accuracy_score(y, pred),
            "F1": f1_score(y, pred, zero_division=0),
            "재현율": recall_score(y, pred, zero_division=0),
            "정밀도": precision_score(y, pred, zero_division=0),
            "AUC": roc_auc_score(y, prob) if prob is not None else np.nan,
        })

    score = pd.DataFrame(rows)
    base = score.loc[score["모델"] == BASELINE, "정확도"].iloc[0]
    score["기준선 대비(%p)"] = (100 * (score["정확도"] - base)).round(1)
    return score, preds


def cut_rule(X, y, folds: int = 5, seed: int = 0,
             weight: str | None = "balanced", model: str = "xgboost") -> float:
    """판정 임계값을 학습 구간만 보고 정한다.

    학습 구간의 양성 비율만큼을 양성으로 예측하도록 자르는 지점을 찾는다.
    검증 성적을 보고 고르면 그 자체가 누수다 — 실제 운용에서는 임계값을
    미리 정해야 하므로, 여기서는 학습 구간 교차검증 확률만 쓴다.

    기본값 0.5 는 쓸 수 없다. 양성이 34% 뿐이라 확률이 0.5 를 거의 넘지 않아
    아무것도 잡지 못한다(F1 0.03).
    """
    X = _frame(X)
    y = np.asarray(y, dtype=int)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    pipe = make_pipeline(impute(X.columns), candidates(seed, weight)[model])
    prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(np.quantile(prob, 1 - y.mean()))


MIN_TRAIN = 60
"""시간순 평가의 최소 학습 표본.

학습 60로트 미만에서는 모델이 전부 양성으로 찍거나(40~43) 아예 작동하지
않는다(44~58). 그 구간을 섞어 평균내면 실제보다 비관적으로 나온다.
"""


def timed(X, y, model: str = "xgboost", weight: str | None = "balanced",
          cut: float | None = None, min_train: int = MIN_TRAIN,
          min_pos: int = 4) -> pd.DataFrame:
    """시간순 평가. 학습 구간을 한 로트씩 늘려가며 나머지로 검증한다.

    입력은 **시간순으로 정렬되어 있어야 한다.**

    cut 을 주지 않으면 분할마다 `cut_rule` 로 새로 정한다.
    고정값을 주면 그 값을 쓰되, 검증을 보고 고른 값이라면 누수임을 유의한다.
    """
    X = _frame(X)
    y = np.asarray(y, dtype=int)
    rows = []
    for n in range(min_train, len(y)):
        tr, te = slice(0, n), slice(n, None)
        if y[te].sum() < min_pos or len(set(y[tr])) < 2:
            continue
        c = cut_rule(X.iloc[tr], y[tr], weight=weight, model=model) if cut is None else cut
        pipe = make_pipeline(impute(X.columns), candidates(0, weight)[model])
        pipe.fit(X.iloc[tr], y[tr])
        prob = pipe.predict_proba(X.iloc[te])[:, 1]
        pred = (prob >= c).astype(int)
        rows.append({
            "학습": n, "검증": len(y) - n, "양성": int(y[te].sum()), "임계값": c,
            "정확도": accuracy_score(y[te], pred),
            "F1": f1_score(y[te], pred, zero_division=0),
            "재현율": recall_score(y[te], pred, zero_division=0),
            "정밀도": precision_score(y[te], pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def confusion(y, pred) -> pd.DataFrame:
    """행이 실제, 열이 예측."""
    cm = confusion_matrix(np.asarray(y, dtype=int), pred)
    return pd.DataFrame(cm, index=["정상 실제", "불량 실제"],
                        columns=["정상 예측", "불량 예측"])


def learning_curve(X, y, sizes=None, repeats: int = 8,
                   model: str = "random forest") -> pd.DataFrame:
    """표본을 줄여가며 같은 측정을 반복한다. 곡선이 아직 오르는 중인지 본다."""
    X = _frame(X)
    y = np.asarray(y, dtype=int)
    n = len(y)
    sizes = [s for s in (sizes or [40, 55, 70, 85, n]) if s <= n]

    rows = []
    for size in sizes:
        acc, f1s, auc = [], [], []
        for seed in range(repeats):
            idx = (np.arange(n) if size == n else
                   train_test_split(np.arange(n), train_size=size,
                                    stratify=y, random_state=seed)[0])
            Xs, ys = X.iloc[idx], y[idx]
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            m = make_pipeline(impute(Xs.columns), candidates(seed)[model])
            pred = cross_val_predict(m, Xs, ys, cv=cv)
            prob = _proba(m, Xs, ys, cv)
            acc.append(accuracy_score(ys, pred))
            f1s.append(f1_score(ys, pred, zero_division=0))
            if prob is not None:
                auc.append(roc_auc_score(ys, prob))
        rows.append({"표본": size, "정확도": np.mean(acc), "편차": np.std(acc),
                     "F1": np.mean(f1s), "AUC": np.mean(auc) if auc else np.nan})
    return pd.DataFrame(rows)


def ceiling(x, y, bins: int = 5) -> float:
    """구간마다 다수 쪽을 찍는 모델의 정확도.

    해당 변수로 도달할 수 있는 상한이다. 표본을 늘려도 이 값을 넘지 못한다.
    """
    s = pd.Series(np.asarray(x, dtype=float))
    d = pd.DataFrame({"y": np.asarray(y, dtype=int),
                      "bin": pd.qcut(s, bins, duplicates="drop")})
    g = d.groupby("bin", observed=True)["y"].agg(["count", "mean"])
    hit = (g["count"] * np.maximum(g["mean"], 1 - g["mean"])).sum()
    return float(hit / g["count"].sum())


def baseline_accuracy(y) -> float:
    """다수 쪽만 찍는 모델의 정확도. 모든 성적은 이 값과 비교해야 한다."""
    y = np.asarray(y, dtype=int)
    return float(max(y.mean(), 1 - y.mean()))
