from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DT_HOURS = 1.0 / 6.0
SMOOTH_WINDOW = 121
MIN_SEGMENT_LENGTH = 500
CV_SPLITS = 5
RIDGE_ALPHAS = np.logspace(-3, 4, 40)
PERSISTENCE_POINTS = 6

CANDIDATES = [
    "降雨量",
    "孔隙水压力",
    "微震事件数",
    "干湿入渗系数",
    "爆破点距离",
    "单段最大药量",
]


@dataclass(frozen=True)
class Paths:
    root: Path
    out: Path
    data: Path
    tables: Path
    figures: Path
    logs: Path


def make_paths(root: Path) -> Paths:
    out = root / "problem5_outputs"
    paths = Paths(
        root=root,
        out=out,
        data=out / "data",
        tables=out / "tables",
        figures=out / "figures",
        logs=out / "logs",
    )
    for path in [paths.out, paths.data, paths.tables, paths.figures, paths.logs]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def locate_preprocessed_attachment5(root: Path) -> Path:
    data_dir = root / "preprocess_outputs" / "data"
    candidates = sorted(data_dir.glob("*.xlsx"), key=lambda p: p.name)
    if len(candidates) < 5:
        raise FileNotFoundError("未找到附件5保守预处理数据。")
    named = [p for p in candidates if "5" in p.name]
    return named[0] if named else candidates[4]


def read_attachment5(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0)
    if raw.shape[1] < 8:
        raise ValueError("附件5字段数量不足，无法进行变量组合建模。")

    time_col = raw.columns[8] if raw.shape[1] > 8 and "解析" in str(raw.columns[8]) else raw.columns[0]
    out = pd.DataFrame(
        {
            "时间": pd.to_datetime(raw[time_col], errors="coerce"),
            "表面位移_mm": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
            "降雨量_mm": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
            "孔隙水压力_kPa": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
            "微震事件数": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
            "干湿入渗系数": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
            "爆破点距离_m": pd.to_numeric(raw.iloc[:, 6], errors="coerce"),
            "单段最大药量_kg": pd.to_numeric(raw.iloc[:, 7], errors="coerce"),
        }
    )
    if "是否爆破" in raw.columns:
        out["是否爆破"] = pd.to_numeric(raw["是否爆破"], errors="coerce").fillna(0).astype(int)
    else:
        out["是否爆破"] = (
            out["爆破点距离_m"].notna() | out["单段最大药量_kg"].notna()
        ).astype(int)
    out["爆破点距离_m_建模"] = out["爆破点距离_m"].fillna(0.0)
    out["单段最大药量_kg_建模"] = out["单段最大药量_kg"].fillna(0.0)
    return out


def check_time_interval(df: pd.DataFrame) -> str:
    diffs = df["时间"].diff().dropna()
    if diffs.empty:
        return "样本不足，无法检查时间间隔"
    mode_delta = diffs.mode().iloc[0]
    abnormal = int((diffs != pd.Timedelta(minutes=10)).sum())
    return f"众数时间间隔={mode_delta}, 非10分钟间隔数量={abnormal}"


def prepare_trend(y: pd.Series, window: int = SMOOTH_WINDOW) -> np.ndarray:
    filled = y.astype(float).interpolate("linear", limit_direction="both")
    trend = (
        filled.rolling(window, center=True, min_periods=max(5, window // 3))
        .median()
        .interpolate("linear", limit_direction="both")
    )
    return trend.to_numpy(dtype=float)


def segment_sse_function(y: np.ndarray):
    n = len(y)
    x = np.arange(n, dtype=float)
    sx = np.r_[0.0, np.cumsum(x)]
    sy = np.r_[0.0, np.cumsum(y)]
    sxx = np.r_[0.0, np.cumsum(x * x)]
    sxy = np.r_[0.0, np.cumsum(x * y)]
    syy = np.r_[0.0, np.cumsum(y * y)]

    def sse(left: int, right: int) -> float:
        m = right - left
        if m <= 1:
            return 0.0
        sum_x = sx[right] - sx[left]
        sum_y = sy[right] - sy[left]
        sum_xx = sxx[right] - sxx[left]
        sum_xy = sxy[right] - sxy[left]
        sum_yy = syy[right] - syy[left]
        den = m * sum_xx - sum_x * sum_x
        if abs(den) < 1e-12:
            return float(max(sum_yy - sum_y * sum_y / m, 0.0))
        slope = (m * sum_xy - sum_x * sum_y) / den
        intercept = (sum_y - slope * sum_x) / m
        value = (
            sum_yy
            + m * intercept * intercept
            + slope * slope * sum_xx
            + 2 * intercept * slope * sum_x
            - 2 * intercept * sum_y
            - 2 * slope * sum_xy
        )
        return float(max(value, 0.0))

    return sse


def best_two_breaks(y: np.ndarray, min_len: int = MIN_SEGMENT_LENGTH) -> tuple[float, int, int]:
    n = len(y)
    sse = segment_sse_function(y)
    best = (float("inf"), None, None)
    for t1 in range(min_len, n - 2 * min_len, 20):
        for t2 in range(t1 + min_len, n - min_len, 20):
            value = sse(0, t1) + sse(t1, t2) + sse(t2, n)
            if value < best[0]:
                best = (value, t1, t2)
    if best[1] is None or best[2] is None:
        raise RuntimeError("无法搜索三阶段转换节点。")

    _, center1, center2 = best
    fine_best = best
    for t1 in range(max(min_len, center1 - 100), min(n - 2 * min_len, center1 + 101)):
        for t2 in range(max(t1 + min_len, center2 - 100), min(n - min_len, center2 + 101)):
            value = sse(0, t1) + sse(t1, t2) + sse(t2, n)
            if value < fine_best[0]:
                fine_best = (value, t1, t2)
    return float(fine_best[0]), int(fine_best[1]), int(fine_best[2])


def identify_stages(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    y = out["表面位移_mm"].astype(float)
    trend = prepare_trend(y)
    sse3, t1, t2 = best_two_breaks(trend)

    stage = np.ones(len(out), dtype=int)
    stage[t1:t2] = 2
    stage[t2:] = 3
    out["稳健趋势位移_mm"] = trend
    out["平滑速度_mm_h"] = pd.Series(trend).diff() / DT_HOURS
    out["阶段标签"] = stage
    out["阶段内序号"] = out.groupby("阶段标签").cumcount()
    stage_size = out.groupby("阶段标签")["阶段标签"].transform("size")
    out["阶段内相对进程"] = out["阶段内序号"] / (stage_size - 1).replace(0, 1)

    sse = segment_sse_function(trend)
    sse1 = sse(0, len(trend))
    rows = []
    for stage_no, (left, right) in enumerate([(0, t1), (t1, t2), (t2, len(out))], start=1):
        segment = out.iloc[left:right]
        duration_h = (right - left - 1) * DT_HOURS if right - left > 1 else 0.0
        delta = float(segment["表面位移_mm"].iloc[-1] - segment["表面位移_mm"].iloc[0])
        speed = delta / duration_h if duration_h > 0 else np.nan
        v = segment["平滑速度_mm_h"].dropna()
        rows.append(
            {
                "阶段编号": stage_no,
                "起始序号": left + 1,
                "终止序号": right,
                "起始时间": segment["时间"].iloc[0],
                "终止时间": segment["时间"].iloc[-1],
                "样本数": right - left,
                "持续时间_h": duration_h,
                "位移增量_mm": delta,
                "阶段平均速度_mm_h": speed,
                "平滑速度中位数_mm_h": float(v.median()) if not v.empty else np.nan,
                "平滑速度75分位_mm_h": float(v.quantile(0.75)) if not v.empty else np.nan,
                "平滑速度90分位_mm_h": float(v.quantile(0.90)) if not v.empty else np.nan,
            }
        )
    stage_table = pd.DataFrame(rows)
    stage_table["三段模型SSE"] = sse3
    stage_table["单段模型SSE"] = sse1
    stage_table["SSE相对下降率"] = 1.0 - sse3 / sse1
    return out, stage_table


def decayed_sum(values: pd.Series, window: int, decay: float = 0.72) -> pd.Series:
    arr = values.fillna(0.0).to_numpy(dtype=float)
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        segment = arr[start : i + 1][::-1]
        weights = decay ** np.arange(len(segment))
        out[i] = float(np.dot(segment, weights))
    return pd.Series(out, index=values.index)


def feature_groups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rain = df["降雨量_mm"].fillna(0.0).clip(lower=0.0)
    pore = df["孔隙水压力_kPa"].interpolate("linear", limit_direction="both")
    micro = df["微震事件数"].fillna(0.0).clip(lower=0.0)
    infil = df["干湿入渗系数"].interpolate("linear", limit_direction="both")
    flag = df["是否爆破"].fillna(0).astype(int)
    dist = df["爆破点距离_m_建模"].astype(float)
    charge = df["单段最大药量_kg_建模"].astype(float)
    inv_dist = np.where(flag > 0, 1.0 / (dist + 1.0), 0.0)
    charge_intensity = np.where(flag > 0, charge, 0.0)
    combined = np.where(flag > 0, charge / np.square(dist + 1.0), 0.0)

    return {
        "降雨量": pd.DataFrame(
            {
                "降雨量_当前": rain,
                "降雨量_1h累计": rain.rolling(6, min_periods=1).sum(),
                "降雨量_6h累计": rain.rolling(36, min_periods=1).sum(),
            }
        ),
        "孔隙水压力": pd.DataFrame(
            {
                "孔隙水压力_当前": pore,
                "孔隙水压力_1h变化": pore.diff(6).fillna(0.0),
                "孔隙水压力_6h均值": pore.rolling(36, min_periods=1).mean(),
            }
        ),
        "微震事件数": pd.DataFrame(
            {
                "微震事件数_当前": micro,
                "微震事件数_1h累计": micro.rolling(6, min_periods=1).sum(),
                "微震事件数_6h累计": micro.rolling(36, min_periods=1).sum(),
            }
        ),
        "干湿入渗系数": pd.DataFrame(
            {
                "干湿入渗系数_当前": infil,
                "干湿入渗系数_1h变化": infil.diff(6).fillna(0.0),
                "干湿入渗系数_6h均值": infil.rolling(36, min_periods=1).mean(),
            }
        ),
        "爆破点距离": pd.DataFrame(
            {
                "爆破标记_距离口径": flag,
                "爆破距离倒数": inv_dist,
                "爆破距离倒数_1h衰减": decayed_sum(pd.Series(inv_dist), 6),
                "爆破距离倒数_6h衰减": decayed_sum(pd.Series(inv_dist), 36),
            }
        ),
        "单段最大药量": pd.DataFrame(
            {
                "爆破标记_药量口径": flag,
                "单段最大药量": charge_intensity,
                "单段最大药量_1h衰减": decayed_sum(pd.Series(charge_intensity), 6),
                "单段最大药量_6h衰减": decayed_sum(pd.Series(charge_intensity), 36),
            }
        ),
        "爆破综合": pd.DataFrame(
            {
                "药量距离综合扰动": combined,
                "药量距离综合扰动_1h衰减": decayed_sum(pd.Series(combined), 6),
                "药量距离综合扰动_6h衰减": decayed_sum(pd.Series(combined), 36),
            }
        ),
    }


def trend_features(df: pd.DataFrame) -> pd.DataFrame:
    p = df["阶段内相对进程"].astype(float)
    stage = df["阶段标签"].astype(int)
    out = pd.DataFrame(index=df.index)
    for stage_no in [1, 2, 3]:
        mask = (stage == stage_no).astype(float)
        out[f"阶段{stage_no}_常数项"] = mask
        out[f"阶段{stage_no}_进程一次项"] = mask * p
        out[f"阶段{stage_no}_进程二次项"] = mask * p * p
        if stage_no >= 2:
            out[f"阶段{stage_no}_进程三次项"] = mask * p * p * p
    return out


def build_model_matrix(df: pd.DataFrame, selected_vars: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    frames = [trend_features(df)]
    feature_group = {col: "阶段趋势" for col in frames[0].columns}
    groups = feature_groups(df)
    for var in selected_vars:
        frame = groups[var]
        frames.append(frame)
        for col in frame.columns:
            feature_group[col] = var
    if "爆破点距离" in selected_vars and "单段最大药量" in selected_vars:
        frame = groups["爆破综合"]
        frames.append(frame)
        for col in frame.columns:
            feature_group[col] = "爆破综合"
    X = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, feature_group


def model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE_mm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_mm": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def stagewise_time_folds(df: pd.DataFrame, splits: int = CV_SPLITS) -> list[np.ndarray]:
    """Build non-random validation folds while keeping every stage represented in training."""
    stage_blocks: dict[int, list[np.ndarray]] = {}
    for stage_no in [1, 2, 3]:
        idx = np.where(df["阶段标签"].to_numpy() == stage_no)[0]
        stage_blocks[stage_no] = [block for block in np.array_split(idx, splits) if len(block) > 0]

    folds = []
    for fold_no in range(splits):
        parts = []
        for blocks in stage_blocks.values():
            if fold_no < len(blocks):
                parts.append(blocks[fold_no])
        if parts:
            folds.append(np.sort(np.concatenate(parts)))
    return folds


def evaluate_feature_matrix(df: pd.DataFrame, X: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], dict[str, float], np.ndarray]:
    y = df["表面位移_mm"].to_numpy(dtype=float)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    model.fit(X, y)
    full_pred = model.predict(X)
    train_metrics = model_metrics(y, full_pred)

    cv_pred = np.full(len(df), np.nan)
    for fold in stagewise_time_folds(df):
        train_idx = np.setdiff1d(np.arange(len(df)), fold)
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y[train_idx])
        cv_pred[fold] = fold_model.predict(X.iloc[fold])
    cv_metrics = model_metrics(y, cv_pred)

    stage_rmse = {}
    for stage_no in [1, 2, 3]:
        mask = df["阶段标签"].to_numpy() == stage_no
        stage_rmse[f"阶段{stage_no}_CV_RMSE_mm"] = float(
            np.sqrt(mean_squared_error(y[mask], cv_pred[mask]))
        )
    return train_metrics, cv_metrics, stage_rmse, cv_pred


def evaluate_combination(df: pd.DataFrame, selected_vars: list[str]) -> tuple[dict[str, object], np.ndarray]:
    X, _ = build_model_matrix(df, selected_vars)
    train_metrics, cv_metrics, stage_rmse, cv_pred = evaluate_feature_matrix(df, X)
    row = {
        "剔除变量": next(var for var in CANDIDATES if var not in selected_vars),
        "纳入变量": "、".join(selected_vars),
        "训练RMSE_mm": train_metrics["RMSE_mm"],
        "训练MAE_mm": train_metrics["MAE_mm"],
        "训练R2": train_metrics["R2"],
        "时间分块CV_RMSE_mm": cv_metrics["RMSE_mm"],
        "时间分块CV_MAE_mm": cv_metrics["MAE_mm"],
        "时间分块CV_R2": cv_metrics["R2"],
        **stage_rmse,
    }
    row["阶段CV_RMSE最大值_mm"] = max(stage_rmse.values())
    return row, cv_pred


def trend_baseline_comparison(df: pd.DataFrame, best_row: pd.Series) -> pd.DataFrame:
    X = trend_features(df)
    train_metrics, cv_metrics, stage_rmse, _ = evaluate_feature_matrix(df, X)
    baseline = {
        "模型": "仅阶段趋势基准模型",
        "训练RMSE_mm": train_metrics["RMSE_mm"],
        "训练MAE_mm": train_metrics["MAE_mm"],
        "训练R2": train_metrics["R2"],
        "时间分块CV_RMSE_mm": cv_metrics["RMSE_mm"],
        "时间分块CV_MAE_mm": cv_metrics["MAE_mm"],
        "时间分块CV_R2": cv_metrics["R2"],
        **stage_rmse,
    }
    best = {
        "模型": "阶段趋势+最优五变量模型",
        "训练RMSE_mm": float(best_row["训练RMSE_mm"]),
        "训练MAE_mm": float(best_row["训练MAE_mm"]),
        "训练R2": float(best_row["训练R2"]),
        "时间分块CV_RMSE_mm": float(best_row["时间分块CV_RMSE_mm"]),
        "时间分块CV_MAE_mm": float(best_row["时间分块CV_MAE_mm"]),
        "时间分块CV_R2": float(best_row["时间分块CV_R2"]),
        "阶段1_CV_RMSE_mm": float(best_row["阶段1_CV_RMSE_mm"]),
        "阶段2_CV_RMSE_mm": float(best_row["阶段2_CV_RMSE_mm"]),
        "阶段3_CV_RMSE_mm": float(best_row["阶段3_CV_RMSE_mm"]),
    }
    table = pd.DataFrame([baseline, best])
    baseline_rmse = float(table.loc[0, "时间分块CV_RMSE_mm"])
    best_rmse = float(table.loc[1, "时间分块CV_RMSE_mm"])
    table["相对基准CV_RMSE改善率"] = np.nan
    if baseline_rmse > 0:
        table.loc[1, "相对基准CV_RMSE改善率"] = (baseline_rmse - best_rmse) / baseline_rmse
    return table


def compare_variable_combinations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for selected in combinations(CANDIDATES, 5):
        row, _ = evaluate_combination(df, list(selected))
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        ["时间分块CV_RMSE_mm", "时间分块CV_MAE_mm", "阶段CV_RMSE最大值_mm"]
    )
    table["排序"] = np.arange(1, len(table) + 1)
    return table


def fit_best_model(df: pd.DataFrame, selected_vars: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, feature_group = build_model_matrix(df, selected_vars)
    y = df["表面位移_mm"].to_numpy(dtype=float)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    model.fit(X, y)
    pred = model.predict(X)

    detail = df.copy()
    detail["最优组合拟合位移_mm"] = pred
    detail["最优组合残差_mm"] = detail["表面位移_mm"] - detail["最优组合拟合位移_mm"]

    ridge = model.named_steps["ridgecv"]
    coef_table = pd.DataFrame(
        {
            "特征名称": X.columns,
            "变量组": [feature_group[c] for c in X.columns],
            "标准化系数": ridge.coef_,
            "系数绝对值": np.abs(ridge.coef_),
        }
    ).sort_values("系数绝对值", ascending=False)
    coef_table["模型截距"] = float(ridge.intercept_)
    coef_table["Ridge_alpha"] = float(ridge.alpha_)

    contrib = (
        coef_table.groupby("变量组", as_index=False)["系数绝对值"]
        .sum()
        .rename(columns={"系数绝对值": "标准化系数绝对值合计"})
        .sort_values("标准化系数绝对值合计", ascending=False)
    )
    total = contrib["标准化系数绝对值合计"].sum()
    contrib["相对贡献占比"] = contrib["标准化系数绝对值合计"] / total if total > 0 else np.nan
    return detail, coef_table, contrib


def warning_thresholds(
    df: pd.DataFrame,
    stage_table: pd.DataFrame,
    persistence_points: int = PERSISTENCE_POINTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    velocity = df["平滑速度_mm_h"].fillna(0.0)
    stage1 = velocity[df["阶段标签"] == 1]
    stage2 = velocity[df["阶段标签"] == 2]
    stage3 = velocity[df["阶段标签"] == 3]

    t1 = float(stage1.quantile(0.75))
    t2 = float(max(stage2.quantile(0.90), (stage2.median() + stage3.median()) / 2.0, t1))
    t3 = float(max(stage3.median(), t2))
    thresholds = pd.DataFrame(
        [
            {
                "预警等级": "正常",
                "速度条件_mm_h": f"v < {t1:.3f}",
                "阈值下界_mm_h": -np.inf,
                "阈值上界_mm_h": t1,
                "持续性判据": "不要求连续超限",
                "设置依据": "低于缓慢阶段平滑速度75分位数",
            },
            {
                "预警等级": "关注",
                "速度条件_mm_h": f"{t1:.3f} <= v < {t2:.3f}",
                "阈值下界_mm_h": t1,
                "阈值上界_mm_h": t2,
                "持续性判据": f"连续不少于{persistence_points}个采样点",
                "设置依据": "超过缓慢阶段常态速度上界",
            },
            {
                "预警等级": "警戒",
                "速度条件_mm_h": f"{t2:.3f} <= v < {t3:.3f}",
                "阈值下界_mm_h": t2,
                "阈值上界_mm_h": t3,
                "持续性判据": f"连续不少于{persistence_points}个采样点",
                "设置依据": "达到加速阶段高分位速度水平",
            },
            {
                "预警等级": "危险",
                "速度条件_mm_h": f"v >= {t3:.3f}",
                "阈值下界_mm_h": t3,
                "阈值上界_mm_h": np.inf,
                "持续性判据": f"连续不少于{persistence_points}个采样点",
                "设置依据": "达到快速形变阶段典型速度水平",
            },
        ]
    )

    code = np.zeros(len(df), dtype=int)
    for threshold, level in [(t1, 1), (t2, 2), (t3, 3)]:
        exceed = (velocity >= threshold).astype(int)
        sustained = exceed.rolling(persistence_points, min_periods=persistence_points).sum() >= persistence_points
        code[sustained.fillna(False).to_numpy()] = level
    names = np.array(["正常", "关注", "警戒", "危险"], dtype=object)
    warning = pd.DataFrame(
        {
            "时间": df["时间"],
            "阶段标签": df["阶段标签"],
            "平滑速度_mm_h": velocity,
            "预警等级编码": code,
            "预警等级": names[code],
        }
    )
    return thresholds, warning


def warning_sensitivity(df: pd.DataFrame, persistence_values: list[int] | None = None) -> pd.DataFrame:
    if persistence_values is None:
        persistence_values = [3, 6, 12]
    rows = []
    for persistence in persistence_values:
        _, warning = warning_thresholds(df, pd.DataFrame(), persistence_points=persistence)
        counts = warning["预警等级"].value_counts().to_dict()
        row = {
            "连续采样点数": persistence,
            "持续时间_h": persistence * DT_HOURS,
            "正常样本数": int(counts.get("正常", 0)),
            "关注样本数": int(counts.get("关注", 0)),
            "警戒样本数": int(counts.get("警戒", 0)),
            "危险样本数": int(counts.get("危险", 0)),
        }
        row["关注及以上样本数"] = row["关注样本数"] + row["警戒样本数"] + row["危险样本数"]
        row["警戒及以上样本数"] = row["警戒样本数"] + row["危险样本数"]
        row["关注及以上占比"] = row["关注及以上样本数"] / len(df)
        row["警戒及以上占比"] = row["警戒及以上样本数"] / len(df)
        rows.append(row)
    return pd.DataFrame(rows)


def make_figures(
    paths: Paths,
    detail: pd.DataFrame,
    stage_table: pd.DataFrame,
    combo: pd.DataFrame,
    thresholds: pd.DataFrame,
    baseline_compare: pd.DataFrame,
    warning_sens: pd.DataFrame,
) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    except Exception:
        pass

    plt.figure(figsize=(10.5, 5.2))
    plt.plot(detail["时间"], detail["表面位移_mm"], linewidth=0.6, alpha=0.55, label="Observed")
    plt.plot(detail["时间"], detail["稳健趋势位移_mm"], linewidth=1.2, label="Robust trend")
    for _, row in stage_table.iloc[:2].iterrows():
        plt.axvline(pd.to_datetime(row["终止时间"]), color="#222222", linestyle="--", linewidth=1.0)
    plt.xlabel("Time")
    plt.ylabel("Surface displacement (mm)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(paths.figures / "问题5_阶段划分图.png", dpi=200)
    plt.close()

    plot_combo = combo.sort_values("时间分块CV_RMSE_mm")
    plt.figure(figsize=(9.5, 5.0))
    plt.bar(plot_combo["剔除变量"], plot_combo["时间分块CV_RMSE_mm"], color="#4d7c9b")
    plt.ylabel("Time-block CV RMSE (mm)")
    plt.xlabel("Excluded variable")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(paths.figures / "问题5_变量组合误差比较.png", dpi=200)
    plt.close()

    t1 = float(thresholds.loc[1, "阈值下界_mm_h"])
    t2 = float(thresholds.loc[2, "阈值下界_mm_h"])
    t3 = float(thresholds.loc[3, "阈值下界_mm_h"])
    plt.figure(figsize=(10.5, 5.2))
    plt.plot(detail["时间"], detail["平滑速度_mm_h"], linewidth=0.7, color="#444444", label="Smoothed velocity")
    for value, label, color in [(t1, "Attention", "#d18f1b"), (t2, "Warning", "#c97028"), (t3, "Danger", "#c43c39")]:
        plt.axhline(value, linestyle="--", linewidth=1.0, color=color, label=f"{label}: {value:.2f}")
    plt.xlabel("Time")
    plt.ylabel("Velocity (mm/h)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(paths.figures / "问题5_速度阈值预警图.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.8, 4.8))
    plt.bar(baseline_compare["模型"], baseline_compare["时间分块CV_RMSE_mm"], color=["#8d99ae", "#4d7c9b"])
    plt.ylabel("Time-block CV RMSE (mm)")
    plt.xlabel("Model")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(paths.figures / "问题5_阶段趋势基准模型对比.png", dpi=200)
    plt.close()

    sens_plot = warning_sens.melt(
        id_vars=["连续采样点数"],
        value_vars=["关注样本数", "警戒样本数", "危险样本数"],
        var_name="预警等级",
        value_name="样本数",
    )
    levels = ["关注样本数", "警戒样本数", "危险样本数"]
    x = np.arange(len(warning_sens))
    width = 0.22
    plt.figure(figsize=(8.8, 4.8))
    for offset, level, color in zip([-width, 0.0, width], levels, ["#d18f1b", "#c97028", "#c43c39"]):
        values = sens_plot[sens_plot["预警等级"] == level]["样本数"].to_numpy()
        plt.bar(x + offset, values, width=width, label=level.replace("样本数", ""), color=color)
    plt.xticks(x, warning_sens["连续采样点数"].astype(str))
    plt.xlabel("Persistence points")
    plt.ylabel("Number of samples")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(paths.figures / "问题5_预警阈值敏感性分析.png", dpi=200)
    plt.close()


def save_outputs(
    paths: Paths,
    detail: pd.DataFrame,
    stage_table: pd.DataFrame,
    combo: pd.DataFrame,
    coef_table: pd.DataFrame,
    contrib: pd.DataFrame,
    thresholds: pd.DataFrame,
    warning: pd.DataFrame,
    warning_summary: pd.DataFrame,
    baseline_compare: pd.DataFrame,
    warning_sens: pd.DataFrame,
    log: dict[str, object],
) -> None:
    detail.to_csv(paths.data / "问题5_建模明细.csv", index=False, encoding="utf-8-sig")
    warning.to_csv(paths.data / "问题5_预警等级明细.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(paths.data / "问题5_建模结果.xlsx") as writer:
        detail.to_excel(writer, sheet_name="建模明细", index=False)
        stage_table.to_excel(writer, sheet_name="阶段划分", index=False)
        combo.to_excel(writer, sheet_name="变量组合误差", index=False)
        coef_table.to_excel(writer, sheet_name="最优模型参数", index=False)
        contrib.to_excel(writer, sheet_name="因素贡献度", index=False)
        thresholds.to_excel(writer, sheet_name="预警阈值", index=False)
        warning_summary.to_excel(writer, sheet_name="预警等级统计", index=False)
        baseline_compare.to_excel(writer, sheet_name="阶段趋势基准对比", index=False)
        warning_sens.to_excel(writer, sheet_name="预警敏感性", index=False)

    stage_table.to_csv(paths.tables / "问题5_阶段划分与平均速度.csv", index=False, encoding="utf-8-sig")
    combo.to_csv(paths.tables / "问题5_变量组合误差比较.csv", index=False, encoding="utf-8-sig")
    coef_table.to_csv(paths.tables / "问题5_最优变量组合模型参数.csv", index=False, encoding="utf-8-sig")
    contrib.to_csv(paths.tables / "问题5_最优模型因素贡献度.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(paths.tables / "问题5_预警阈值表.csv", index=False, encoding="utf-8-sig")
    warning_summary.to_csv(paths.tables / "问题5_预警等级统计.csv", index=False, encoding="utf-8-sig")
    baseline_compare.to_csv(paths.tables / "问题5_阶段趋势基准模型对比.csv", index=False, encoding="utf-8-sig")
    warning_sens.to_csv(paths.tables / "问题5_预警阈值敏感性分析.csv", index=False, encoding="utf-8-sig")

    with (paths.logs / "问题5_处理日志.json").open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)
    with (paths.logs / "问题5_处理日志.txt").open("w", encoding="utf-8") as f:
        for key, value in log.items():
            f.write(f"{key}: {value}\n")


def main() -> None:
    root = Path(__file__).resolve().parent
    paths = make_paths(root)
    source = locate_preprocessed_attachment5(root)
    raw = read_attachment5(source)
    staged, stage_table = identify_stages(raw)
    combo = compare_variable_combinations(staged)
    best_row = combo.iloc[0]
    best_vars = str(best_row["纳入变量"]).split("、")
    baseline_compare = trend_baseline_comparison(staged, best_row)
    detail, coef_table, contrib = fit_best_model(staged, best_vars)
    thresholds, warning = warning_thresholds(detail, stage_table)
    warning_sens = warning_sensitivity(detail)
    detail = detail.merge(warning[["时间", "预警等级编码", "预警等级"]], on="时间", how="left")
    warning_summary = (
        warning.groupby(["预警等级编码", "预警等级"], as_index=False)
        .size()
        .rename(columns={"size": "样本数"})
        .sort_values("预警等级编码")
    )
    warning_summary["样本占比"] = warning_summary["样本数"] / len(warning)
    make_figures(paths, detail, stage_table, combo, thresholds, baseline_compare, warning_sens)

    best_metrics = model_metrics(
        detail["表面位移_mm"].to_numpy(dtype=float),
        detail["最优组合拟合位移_mm"].to_numpy(dtype=float),
    )
    log = {
        "数据源": str(source),
        "样本数": int(len(raw)),
        "时间范围": [raw["时间"].min(), raw["时间"].max()],
        "时间间隔检查": check_time_interval(raw),
        "爆破记录数": int(raw["是否爆破"].sum()),
        "爆破变量空值口径": "爆破点距离和单段最大药量为空解释为非爆破时刻，建模时编码为0扰动。",
        "阶段识别方法": "稳健中位数平滑后的位移趋势三段线性SSE最小化，最小阶段长度为500个采样点。",
        "阶段划分": stage_table.to_dict(orient="records"),
        "变量组合比较方法": "六类候选变量每次剔除一类，采用相同分阶段趋势项和Ridge残差解释框架，并用5折时间分块交叉验证评价。",
        "最优剔除变量": best_row["剔除变量"],
        "最优纳入变量": best_vars,
        "最优组合训练拟合指标": best_metrics,
        "最优组合时间分块CV指标": {
            "RMSE_mm": float(best_row["时间分块CV_RMSE_mm"]),
            "MAE_mm": float(best_row["时间分块CV_MAE_mm"]),
            "R2": float(best_row["时间分块CV_R2"]),
        },
        "仅阶段趋势基准模型对比": baseline_compare.to_dict(orient="records"),
        "预警阈值口径": "依据附件5自身平滑速度分布设置阈值；关注阈值取第一阶段75分位，警戒阈值取第二阶段90分位与二三阶段中位速度分界的较大值，危险阈值取第三阶段中位速度并保持单调。",
        "预警持续性判据_采样点": PERSISTENCE_POINTS,
        "预警阈值敏感性分析": warning_sens.to_dict(orient="records"),
        "输出目录": str(paths.out),
    }
    save_outputs(
        paths,
        detail,
        stage_table,
        combo,
        coef_table,
        contrib,
        thresholds,
        warning,
        warning_summary,
        baseline_compare,
        warning_sens,
        log,
    )

    print("问题五建模完成")
    print("阶段划分：")
    print(stage_table[["阶段编号", "起始时间", "终止时间", "阶段平均速度_mm_h"]].to_string(index=False))
    print("变量组合误差比较：")
    print(combo[["排序", "剔除变量", "时间分块CV_RMSE_mm", "时间分块CV_MAE_mm", "时间分块CV_R2"]].to_string(index=False))
    print("最优组合：", "、".join(best_vars), "；剔除变量：", best_row["剔除变量"])
    print("阶段趋势基准模型对比：")
    print(baseline_compare[["模型", "时间分块CV_RMSE_mm", "时间分块CV_MAE_mm", "时间分块CV_R2", "相对基准CV_RMSE改善率"]].to_string(index=False))
    print("预警阈值：")
    print(thresholds[["预警等级", "速度条件_mm_h", "持续性判据"]].to_string(index=False))
    print("预警阈值敏感性分析：")
    print(warning_sens.to_string(index=False))


if __name__ == "__main__":
    main()
