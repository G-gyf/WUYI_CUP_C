from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DT_HOURS = 1.0 / 6.0
MAIN_SMOOTH_WINDOW = 61
MIN_SEGMENT_LENGTH = 500
CV_SPLITS = 5
RIDGE_ALPHAS = np.logspace(-3, 4, 40)
TARGET_TIMES = [
    "2025-05-09 12:00",
    "2025-05-27 08:00",
    "2025-06-01 12:00",
    "2025-06-03 22:00",
    "2025-06-04 01:40",
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
    out = root / "problem4_outputs"
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


def locate_preprocessed_attachment4(root: Path) -> Path:
    data_dir = root / "preprocess_outputs" / "data"
    candidates = [
        p
        for p in data_dir.glob("*.xlsx")
        if ("附件4" in p.name or "问题4" in p.name) and "保守预处理" in p.name
    ]
    if not candidates:
        candidates = sorted(data_dir.glob("*.xlsx"), key=lambda x: x.name)
        if len(candidates) < 4:
            raise FileNotFoundError("未找到附件4保守预处理文件。")
        return candidates[3]
    return sorted(candidates, key=lambda x: (len(str(x)), x.name))[0]


def read_attachment4(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_excel(path, sheet_name="训练集")
    test = pd.read_excel(path, sheet_name="实验集")
    for df in [train, test]:
        if "时间解析" in df.columns:
            df["时间解析"] = pd.to_datetime(df["时间解析"])
        else:
            df["时间解析"] = pd.to_datetime(df["时间"])
    return train, test


def check_time_interval(df: pd.DataFrame) -> str:
    diffs = df["时间解析"].diff().dropna()
    if diffs.empty:
        return "样本不足，无法检查时间间隔"
    mode_delta = diffs.mode().iloc[0]
    abnormal = int((diffs != pd.Timedelta(minutes=10)).sum())
    return f"众数时间间隔={mode_delta}, 非10分钟间隔数量={abnormal}"


def prepare_trend(y: np.ndarray, window: int = MAIN_SMOOTH_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    series = pd.Series(y).astype(float)
    interpolated = series.interpolate("linear", limit_direction="both")
    trend = (
        interpolated.rolling(window, center=True, min_periods=max(5, window // 3))
        .median()
        .interpolate("linear", limit_direction="both")
    )
    return interpolated.to_numpy(dtype=float), trend.to_numpy(dtype=float)


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
        raise RuntimeError("无法搜索训练集三阶段转换节点。")

    _, center1, center2 = best
    fine_best = best
    left1 = max(min_len, center1 - 100)
    right1 = min(n - 2 * min_len, center1 + 101)
    for t1 in range(left1, right1):
        left2 = max(t1 + min_len, center2 - 100)
        right2 = min(n - min_len, center2 + 101)
        for t2 in range(left2, right2):
            value = sse(0, t1) + sse(t1, t2) + sse(t2, n)
            if value < fine_best[0]:
                fine_best = (value, t1, t2)
    return float(fine_best[0]), int(fine_best[1]), int(fine_best[2])


def identify_train_stages(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = train.copy()
    y = out["表面位移_mm"].to_numpy(dtype=float)
    _, trend = prepare_trend(y)
    sse3, t1, t2 = best_two_breaks(trend)
    stage = np.ones(len(out), dtype=int)
    stage[t1:t2] = 2
    stage[t2:] = 3
    out["阶段标签"] = stage
    out["稳健趋势位移_mm"] = trend
    out["滚动72点速度_mm_h"] = pd.Series(trend).diff(72) / (72 * DT_HOURS)
    out["滚动72点加速度_mm_h2"] = out["滚动72点速度_mm_h"].diff(72) / (72 * DT_HOURS)

    sse = segment_sse_function(trend)
    sse1 = sse(0, len(trend))
    rows = []
    for stage_no, (left, right) in enumerate([(0, t1), (t1, t2), (t2, len(out))], start=1):
        segment = out.iloc[left:right]
        duration_h = (right - left - 1) * DT_HOURS if right - left > 1 else 0.0
        avg_speed = (
            (segment["表面位移_mm"].iloc[-1] - segment["表面位移_mm"].iloc[0]) / duration_h
            if duration_h > 0
            else np.nan
        )
        rows.append(
            {
                "阶段编号": stage_no,
                "起始序号": left + 1,
                "终止序号": right,
                "起始时间": segment["时间解析"].iloc[0],
                "终止时间": segment["时间解析"].iloc[-1],
                "样本数": right - left,
                "阶段平均速度_mm_h": avg_speed,
            }
        )
    stage_table = pd.DataFrame(rows)
    stage_table["三段模型SSE"] = sse3
    stage_table["单段模型SSE"] = sse1
    stage_table["SSE相对下降率"] = 1.0 - sse3 / sse1
    return out, stage_table


def add_stage_progress(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["阶段内序号"] = out.groupby("阶段标签").cumcount()
    stage_sizes = out.groupby("阶段标签")["阶段标签"].transform("size")
    denom = (stage_sizes - 1).replace(0, 1)
    out["阶段内相对进程"] = out["阶段内序号"] / denom
    out["阶段进程步长"] = 1.0 / denom
    out["阶段内时间_h"] = out["阶段内序号"] * DT_HOURS
    return out


def decayed_effect(values: pd.Series, horizon: int = 6, tau: float = 3.0) -> pd.Series:
    arr = values.fillna(0.0).to_numpy(dtype=float)
    result = np.zeros_like(arr, dtype=float)
    weights = np.exp(-np.arange(horizon + 1) / tau)
    for lag, weight in enumerate(weights):
        if lag == 0:
            result += weight * arr
        else:
            result[lag:] += weight * arr[:-lag]
    return pd.Series(result, index=values.index)


def build_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    out = add_stage_progress(df)
    out["是否爆破"] = out["是否爆破"].fillna(0).astype(int)
    out["爆破距离_建模"] = out["爆破点距离_m"].fillna(0.0)
    out["单段最大药量_建模"] = out["单段最大药量_kg"].fillna(0.0)
    distance = out["爆破点距离_m"].astype(float)
    charge = out["单段最大药量_kg"].astype(float)
    intensity = (charge / np.maximum(distance, 0.1) ** 2).replace([np.inf, -np.inf], np.nan)
    out["爆破强度"] = intensity.fillna(0.0)
    out["爆破强度_1h衰减"] = decayed_effect(out["爆破强度"], horizon=6, tau=3.0)

    out["降雨量_1h累计"] = out["降雨量_mm"].rolling(6, min_periods=1).sum()
    out["降雨量_6h累计"] = out["降雨量_mm"].rolling(36, min_periods=1).sum()
    out["微震事件数_1h累计"] = out["微震事件数"].rolling(6, min_periods=1).sum()
    out["微震事件数_6h累计"] = out["微震事件数"].rolling(36, min_periods=1).sum()
    out["阶段进程平方"] = out["阶段内相对进程"] ** 2

    if is_train:
        out["位移增量_mm"] = out["表面位移_mm"].diff()
        out.loc[out.index[0], "位移增量_mm"] = np.nan
        out["阶段归一化位移斜率_mm"] = out["位移增量_mm"] / out["阶段进程步长"]
    return out


BASE_FEATURE_COLS = [
    "阶段内相对进程",
    "阶段进程平方",
    "降雨量_mm",
    "降雨量_1h累计",
    "降雨量_6h累计",
    "孔隙水压力_kPa",
    "微震事件数",
    "微震事件数_1h累计",
    "微震事件数_6h累计",
    "是否爆破",
    "爆破强度",
    "爆破强度_1h衰减",
]
FEATURE_COLS = [f"{col}_模型输入" for col in BASE_FEATURE_COLS]

FEATURE_GROUPS = {
    "阶段演化": ["阶段内相对进程_模型输入", "阶段进程平方_模型输入"],
    "降雨": ["降雨量_mm_模型输入", "降雨量_1h累计_模型输入", "降雨量_6h累计_模型输入"],
    "孔隙水压力": ["孔隙水压力_kPa_模型输入"],
    "微震": ["微震事件数_模型输入", "微震事件数_1h累计_模型输入", "微震事件数_6h累计_模型输入"],
    "爆破": ["是否爆破_模型输入", "爆破强度_模型输入", "爆破强度_1h衰减_模型输入"],
}


def add_model_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in BASE_FEATURE_COLS:
        out[f"{col}_模型输入"] = out[col]
    return out


def clip_test_model_inputs(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = test.copy()
    rows = []
    for stage_no in [1, 2, 3]:
        train_stage = train[train["阶段标签"] == stage_no]
        test_mask = out["阶段标签"] == stage_no
        for base_col in BASE_FEATURE_COLS:
            model_col = f"{base_col}_模型输入"
            lower = float(train_stage[model_col].min())
            upper = float(train_stage[model_col].max())
            before = out.loc[test_mask, model_col].copy()
            after = before.clip(lower=lower, upper=upper)
            changed = int((before != after).sum())
            out.loc[test_mask, model_col] = after
            rows.append(
                {
                    "阶段编号": stage_no,
                    "变量": base_col,
                    "训练集下界": lower,
                    "训练集上界": upper,
                    "实验集被截尾数量": changed,
                }
            )
    return out, pd.DataFrame(rows)


def add_stage_trend(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()
    train_out["阶段趋势位移_mm"] = np.nan
    test_out["阶段趋势位移_mm"] = np.nan
    rows = []

    for stage_no in [1, 2, 3]:
        train_mask = train_out["阶段标签"] == stage_no
        test_mask = test_out["阶段标签"] == stage_no
        degree = 1 if stage_no == 1 else 2
        x_train = train_out.loc[train_mask, "阶段内相对进程"].to_numpy(dtype=float)
        y_train = train_out.loc[train_mask, "表面位移_mm"].to_numpy(dtype=float)
        coef = np.polyfit(x_train, y_train, degree)
        train_trend = np.polyval(coef, x_train)
        test_trend = np.polyval(coef, test_out.loc[test_mask, "阶段内相对进程"].to_numpy(dtype=float))
        train_out.loc[train_mask, "阶段趋势位移_mm"] = train_trend
        test_out.loc[test_mask, "阶段趋势位移_mm"] = test_trend
        pred_train = train_trend
        metrics = model_metrics(y_train, pred_train, unit="mm")
        rows.append(
            {
                "阶段编号": stage_no,
                "趋势多项式阶数": degree,
                "系数_高次到常数": ";".join([f"{v:.12g}" for v in coef]),
                **metrics,
            }
        )

    train_out["阶段趋势增量_mm"] = train_out.groupby("阶段标签")["阶段趋势位移_mm"].diff()
    test_out["阶段趋势增量_mm"] = test_out.groupby("阶段标签")["阶段趋势位移_mm"].diff().fillna(0.0)
    train_out["阶段归一化残差斜率_mm"] = (
        (train_out["位移增量_mm"] - train_out["阶段趋势增量_mm"]) / train_out["阶段进程步长"]
    )
    return train_out, test_out, pd.DataFrame(rows)


def model_metrics(y_true: np.ndarray, y_pred: np.ndarray, unit: str = "mm") -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan
    return {f"RMSE_{unit}": rmse, f"MAE_{unit}": mae, f"Bias_{unit}": bias, "R2": r2}


def fit_models_by_stage(train: pd.DataFrame) -> tuple[dict[int, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_defs = {
        "线性回归": make_pipeline(StandardScaler(), LinearRegression()),
        "岭回归": make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS)),
        "Huber稳健回归": make_pipeline(StandardScaler(), HuberRegressor(alpha=0.001, max_iter=1000)),
        "随机森林辅助对照": RandomForestRegressor(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=20,
            random_state=2026,
            n_jobs=-1,
        ),
    }
    main_models: dict[int, object] = {}
    metric_rows = []
    cv_rows = []
    coef_rows = []

    for stage_no in [1, 2, 3]:
        stage_df = train[(train["阶段标签"] == stage_no) & train["阶段归一化残差斜率_mm"].notna()].copy()
        x = stage_df[FEATURE_COLS].to_numpy(dtype=float)
        y = stage_df["阶段归一化残差斜率_mm"].to_numpy(dtype=float)
        progress_step = stage_df["阶段进程步长"].to_numpy(dtype=float)
        true_delta = stage_df["位移增量_mm"].to_numpy(dtype=float)
        trend_delta = stage_df["阶段趋势增量_mm"].to_numpy(dtype=float)

        for model_name, model in model_defs.items():
            fitted = clone(model)
            fitted.fit(x, y)
            pred_slope = fitted.predict(x)
            pred_delta = trend_delta + pred_slope * progress_step
            row = {"阶段编号": stage_no, "模型": model_name, "样本数": len(stage_df)}
            row.update(model_metrics(true_delta, pred_delta, unit="delta_mm"))
            metric_rows.append(row)
            if model_name == "岭回归":
                main_models[stage_no] = fitted

        cv_pred = np.full(len(stage_df), np.nan)
        indices = np.arange(len(stage_df))
        folds = np.array_split(indices, CV_SPLITS)
        for fold_no, valid_idx in enumerate(folds, start=1):
            train_idx = np.setdiff1d(indices, valid_idx)
            if len(valid_idx) == 0 or len(train_idx) < 20:
                continue
            model = clone(model_defs["岭回归"])
            model.fit(x[train_idx], y[train_idx])
            cv_pred[valid_idx] = trend_delta[valid_idx] + model.predict(x[valid_idx]) * progress_step[valid_idx]
            fold_row = {"阶段编号": stage_no, "折号": fold_no, "验证样本数": len(valid_idx)}
            fold_row.update(model_metrics(true_delta[valid_idx], cv_pred[valid_idx], unit="delta_mm"))
            cv_rows.append(fold_row)

        ridge = main_models[stage_no]
        coef = ridge.named_steps["ridgecv"].coef_
        for col, value in zip(FEATURE_COLS, coef):
            coef_rows.append({"阶段编号": stage_no, "变量": col, "标准化系数": float(value)})

    return main_models, pd.DataFrame(metric_rows), pd.DataFrame(cv_rows), pd.DataFrame(coef_rows)


def contribution_table(coef_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage_no in sorted(coef_table["阶段编号"].unique()):
        stage_coef = coef_table[coef_table["阶段编号"] == stage_no]
        total = 0.0
        group_values = {}
        for group, cols in FEATURE_GROUPS.items():
            value = float(stage_coef.loc[stage_coef["变量"].isin(cols), "标准化系数"].abs().sum())
            group_values[group] = value
            total += value
        for group, value in group_values.items():
            rows.append(
                {
                    "阶段编号": stage_no,
                    "因素类别": group,
                    "系数绝对值合计": value,
                    "归一化贡献度": value / total if total > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def predict_train_and_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    models: dict[int, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    test_out = test.copy()

    train_out["岭回归预测增量_mm"] = np.nan
    for stage_no, model in models.items():
        mask = (train_out["阶段标签"] == stage_no) & train_out["阶段归一化残差斜率_mm"].notna()
        pred_slope = model.predict(
            train_out.loc[mask, FEATURE_COLS].to_numpy(dtype=float)
        )
        train_out.loc[mask, "岭回归预测增量_mm"] = (
            train_out.loc[mask, "阶段趋势增量_mm"].to_numpy(dtype=float)
            + pred_slope * train_out.loc[mask, "阶段进程步长"].to_numpy(dtype=float)
        )
    train_out["岭回归累加拟合位移_mm"] = train_out["表面位移_mm"].iloc[0]
    valid_delta = train_out["岭回归预测增量_mm"].fillna(0.0)
    train_out["岭回归累加拟合位移_mm"] = train_out["表面位移_mm"].iloc[0] + valid_delta.cumsum()

    test_out["岭回归预测增量_原始_mm"] = np.nan
    for stage_no, model in models.items():
        mask = test_out["阶段标签"] == stage_no
        pred_slope = model.predict(
            test_out.loc[mask, FEATURE_COLS].to_numpy(dtype=float)
        )
        residual_delta = pred_slope * test_out.loc[mask, "阶段进程步长"].to_numpy(dtype=float)
        residual_delta = residual_delta - np.nanmean(residual_delta)
        test_out.loc[mask, "岭回归预测增量_原始_mm"] = (
            test_out.loc[mask, "阶段趋势增量_mm"].to_numpy(dtype=float)
            + residual_delta
        )
    test_out["岭回归预测增量_mm"] = test_out["岭回归预测增量_原始_mm"]
    deltas = test_out["岭回归预测增量_mm"].fillna(0.0).to_numpy(dtype=float)
    deltas[0] = 0.0
    test_out["表面位移预测值_mm"] = np.maximum(train_out["表面位移_mm"].iloc[0] + np.cumsum(deltas), 0.0)
    return train_out, test_out


def table_4_1(test_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lookup = test_pred.set_index("时间解析")
    for text in TARGET_TIMES:
        time_value = pd.Timestamp(text)
        if time_value not in lookup.index:
            nearest_pos = int(np.argmin(np.abs((test_pred["时间解析"] - time_value).dt.total_seconds())))
            row = test_pred.iloc[nearest_pos]
            match_type = "最近时刻"
        else:
            row = lookup.loc[time_value]
            match_type = "精确匹配"
        rows.append(
            {
                "时间点": time_value,
                "表面位移预测值_mm": float(row["表面位移预测值_mm"]),
                "阶段标签": int(row["阶段标签"]),
                "匹配方式": match_type,
            }
        )
    return pd.DataFrame(rows)


def make_figures(paths: Paths, train_pred: pd.DataFrame, test_pred: pd.DataFrame, stage_table: pd.DataFrame) -> None:
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10.5, 5.2))
    plt.plot(train_pred["时间解析"], train_pred["表面位移_mm"], linewidth=0.7, alpha=0.45, label="Observed")
    plt.plot(train_pred["时间解析"], train_pred["稳健趋势位移_mm"], linewidth=1.3, label="Robust trend")
    for _, row in stage_table.iloc[:2].iterrows():
        plt.axvline(pd.to_datetime(row["终止时间"]), color="#222222", linestyle="--", linewidth=1.0)
    plt.xlabel("Time")
    plt.ylabel("Surface displacement (mm)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(paths.figures / "问题4_训练集阶段划分图.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10.5, 5.2))
    plt.plot(train_pred["时间解析"], train_pred["表面位移_mm"], linewidth=0.7, label="Observed")
    plt.plot(train_pred["时间解析"], train_pred["岭回归累加拟合位移_mm"], linewidth=0.9, label="Cumulative fitted")
    plt.xlabel("Time")
    plt.ylabel("Surface displacement (mm)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(paths.figures / "问题4_训练集累加拟合图.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10.5, 5.2))
    for stage_no, color in [(1, "#2f6f9f"), (2, "#d18f1b"), (3, "#c43c39")]:
        mask = test_pred["阶段标签"] == stage_no
        plt.plot(
            test_pred.loc[mask, "时间解析"],
            test_pred.loc[mask, "表面位移预测值_mm"],
            linewidth=1.2,
            color=color,
            label=f"Stage {stage_no}",
        )
    plt.xlabel("Time")
    plt.ylabel("Predicted surface displacement (mm)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(paths.figures / "问题4_实验集表面位移预测图.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10.5, 5.2))
    plt.plot(test_pred["时间解析"], test_pred["岭回归预测增量_mm"], linewidth=0.8, color="#444444")
    blast = test_pred["是否爆破"].astype(bool)
    if blast.any():
        plt.scatter(
            test_pred.loc[blast, "时间解析"],
            test_pred.loc[blast, "岭回归预测增量_mm"],
            s=16,
            color="#c43c39",
            label="Blasting event",
        )
        plt.legend(frameon=False)
    plt.xlabel("Time")
    plt.ylabel("Predicted displacement increment (mm)")
    plt.tight_layout()
    plt.savefig(paths.figures / "问题4_实验集预测增量与爆破标记图.png", dpi=200)
    plt.close()


def save_outputs(
    paths: Paths,
    train_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    stage_table: pd.DataFrame,
    metrics: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    trend_table: pd.DataFrame,
    coef_table: pd.DataFrame,
    contrib: pd.DataFrame,
    clip_table: pd.DataFrame,
    table41: pd.DataFrame,
    log: dict[str, object],
) -> None:
    train_pred.to_csv(paths.data / "问题4_训练集分阶段建模明细.csv", index=False, encoding="utf-8-sig")
    test_pred.to_csv(paths.data / "问题4_实验集表面位移预测明细.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(paths.data / "问题4_建模与预测结果.xlsx") as writer:
        train_pred.to_excel(writer, sheet_name="训练集建模明细", index=False)
        test_pred.to_excel(writer, sheet_name="实验集预测明细", index=False)
        stage_table.to_excel(writer, sheet_name="训练集阶段划分", index=False)
        table41.to_excel(writer, sheet_name="表4_1", index=False)
        trend_table.to_excel(writer, sheet_name="阶段趋势模型", index=False)
        metrics.to_excel(writer, sheet_name="模型检验指标", index=False)
        cv_metrics.to_excel(writer, sheet_name="时间分块验证", index=False)
        coef_table.to_excel(writer, sheet_name="岭回归系数", index=False)
        contrib.to_excel(writer, sheet_name="因素贡献度", index=False)
        clip_table.to_excel(writer, sheet_name="实验集特征截尾", index=False)

    stage_table.to_csv(paths.tables / "问题4_训练集阶段划分.csv", index=False, encoding="utf-8-sig")
    trend_table.to_csv(paths.tables / "问题4_阶段趋势模型参数.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(paths.tables / "问题4_模型检验指标.csv", index=False, encoding="utf-8-sig")
    cv_metrics.to_csv(paths.tables / "问题4_时间分块交叉验证.csv", index=False, encoding="utf-8-sig")
    coef_table.to_csv(paths.tables / "问题4_岭回归标准化系数.csv", index=False, encoding="utf-8-sig")
    contrib.to_csv(paths.tables / "问题4_因素贡献度.csv", index=False, encoding="utf-8-sig")
    clip_table.to_csv(paths.tables / "问题4_实验集特征截尾记录.csv", index=False, encoding="utf-8-sig")
    table41.to_csv(paths.tables / "表4_1_实验集指定时刻预测值.csv", index=False, encoding="utf-8-sig")

    with (paths.logs / "问题4_处理日志.json").open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)
    with (paths.logs / "问题4_处理日志.txt").open("w", encoding="utf-8") as f:
        for key, value in log.items():
            f.write(f"{key}: {value}\n")


def main() -> None:
    root = Path(__file__).resolve().parent
    paths = make_paths(root)
    source = locate_preprocessed_attachment4(root)
    train_raw, test_raw = read_attachment4(source)

    train_stage, stage_table = identify_train_stages(train_raw)
    train_features = add_model_input_columns(build_features(train_stage, is_train=True))
    test_features = add_model_input_columns(build_features(test_raw, is_train=False))
    test_features, clip_table = clip_test_model_inputs(train_features, test_features)
    train_features, test_features, trend_table = add_stage_trend(train_features, test_features)

    models, metrics, cv_metrics, coef_table = fit_models_by_stage(train_features)
    contrib = contribution_table(coef_table)
    train_pred, test_pred = predict_train_and_test(train_features, test_features, models)
    table41 = table_4_1(test_pred)
    make_figures(paths, train_pred, test_pred, stage_table)

    train_disp_metrics = model_metrics(
        train_pred["表面位移_mm"].to_numpy(dtype=float),
        train_pred["岭回归累加拟合位移_mm"].to_numpy(dtype=float),
        unit="mm",
    )
    log = {
        "数据源": str(source),
        "训练集样本数": int(len(train_raw)),
        "实验集样本数": int(len(test_raw)),
        "训练集时间检查": check_time_interval(train_raw),
        "实验集时间检查": check_time_interval(test_raw),
        "训练集爆破记录数": int(train_raw["是否爆破"].fillna(0).astype(int).sum()),
        "实验集爆破记录数": int(test_raw["是否爆破"].fillna(0).astype(int).sum()),
        "训练集阶段划分": stage_table.to_dict(orient="records"),
        "实验集阶段分布": test_raw["阶段标签"].value_counts().sort_index().to_dict(),
        "实验集建模特征截尾数量": int(clip_table["实验集被截尾数量"].sum()),
        "主模型": "分阶段岭回归",
        "预测目标": "阶段归一化位移斜率；按各阶段进程步长换算为10分钟增量后累加为表面位移；增量不截断，累计位移施加非负下界",
        "模型分解": "阶段趋势项 + 趋势外残差增量项；实验集残差增量在各阶段内中心化，仅修正局部形态",
        "实验集初始位移基准_mm": float(train_raw["表面位移_mm"].iloc[0]),
        "累加拟合位移误差": train_disp_metrics,
        "表4_1": table41.to_dict(orient="records"),
        "输出目录": str(paths.out),
    }
    save_outputs(
        paths,
        train_pred,
        test_pred,
        stage_table,
        metrics,
        cv_metrics,
        trend_table,
        coef_table,
        contrib,
        clip_table,
        table41,
        log,
    )

    print("问题四建模完成")
    print(f"训练集阶段转换时间: {stage_table.loc[0, '终止时间']} ; {stage_table.loc[1, '终止时间']}")
    print("表4.1预测值:")
    print(table41.to_string(index=False))
    print("训练集累加拟合误差:", train_disp_metrics)


if __name__ == "__main__":
    main()
