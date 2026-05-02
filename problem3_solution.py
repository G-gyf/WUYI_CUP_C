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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VAR_NAMES = {
    "a": "a_降雨量_mm",
    "b": "b_孔隙水压力_kPa",
    "c": "c_微震事件数",
    "d": "d_深部位移_mm",
    "e": "e_表面位移_mm",
}
FEATURE_KEYS = ["a", "b", "c", "d"]
TARGET_KEY = "e"
ROLLING_SHORT = 7
ROLLING_MEDIAN_WINDOW = 11
ANOMALY_WINDOW = 31
CV_SPLITS = 5
RIDGE_ALPHAS = np.logspace(-3, 4, 40)
LAG_STEPS = [1, 3, 6, 12, 24]


@dataclass(frozen=True)
class Paths:
    root: Path
    out: Path
    data: Path
    tables: Path
    figures: Path
    logs: Path


def make_paths(root: Path) -> Paths:
    out = root / "problem3_outputs"
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


def locate_preprocessed_attachment3(root: Path) -> Path:
    path = root / "preprocess_outputs" / "data" / "附件3_保守预处理.xlsx"
    if path.exists():
        return path
    candidates = [
        p
        for p in root.rglob("*.xlsx")
        if "preprocess_outputs" in str(p) and ("附件3" in p.name or "问题3" in p.name)
    ]
    if not candidates:
        raise FileNotFoundError("未找到附件3保守预处理文件。")
    return sorted(candidates, key=lambda p: (len(str(p)), p.name))[0]


def rolling_median(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=min_periods).median()


def robust_scale(values: pd.Series) -> float:
    valid = values.dropna().to_numpy(dtype=float)
    if valid.size == 0:
        return 1.0
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.nanstd(valid))
    return max(scale, 1e-9)


def centered_local_median_fill(series: pd.Series, window: int = ROLLING_SHORT) -> pd.Series:
    filled = series.copy().astype(float)
    local = rolling_median(filled, window=window)
    filled = filled.fillna(local)
    filled = filled.ffill().bfill()
    return filled


def continuous_interpolate(series: pd.Series) -> pd.Series:
    filled = series.copy().astype(float)
    try:
        filled = filled.interpolate(method="pchip", limit_direction="both")
    except Exception:
        filled = filled.interpolate(method="linear", limit_direction="both")
    filled = filled.ffill().bfill()
    return filled


def gentle_hampel_denoise(series: pd.Series, value_floor: float | None = None) -> tuple[pd.Series, pd.Series]:
    y = series.copy().astype(float)
    med = rolling_median(y, window=ROLLING_MEDIAN_WINDOW)
    residual = y - med
    local_mad = (
        residual.abs()
        .rolling(window=ANOMALY_WINDOW, center=True, min_periods=5)
        .median()
        .fillna(residual.abs().median())
    )
    local_scale = 1.4826 * local_mad
    global_scale = robust_scale(residual)
    threshold = np.maximum(6.0 * local_scale.to_numpy(dtype=float), 4.0 * global_scale)
    spike_mask = residual.abs().to_numpy(dtype=float) > threshold

    denoised = y.copy()
    denoised.loc[spike_mask] = med.loc[spike_mask]
    if value_floor is not None:
        denoised = denoised.clip(lower=value_floor)
    return denoised, pd.Series(spike_mask, index=series.index)


def prepare_one_frame(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    out = df.copy()
    for key, col in VAR_NAMES.items():
        if col not in out.columns:
            continue
        if key == "e" and not is_train:
            out[f"{key}_原始缺失"] = out[col].isna()
            out[f"{key}_补齐值"] = np.nan
            out[f"{key}_建模值"] = np.nan
            out[f"{key}_是否插补"] = out[col].isna()
            continue
        out[f"{key}_原始缺失"] = out[col].isna()

        if key == "a":
            filled = centered_local_median_fill(out[col]).clip(lower=0.0)
            model = filled.copy()
        elif key == "c":
            filled = centered_local_median_fill(out[col]).round().clip(lower=0.0)
            model = filled.copy()
        else:
            filled = continuous_interpolate(out[col])
            model, _ = gentle_hampel_denoise(filled, value_floor=None)

        out[f"{key}_补齐值"] = filled
        out[f"{key}_建模值"] = model
        out[f"{key}_是否插补"] = out[col].isna()

    return out


def rain_anomaly(series: pd.Series, missing_mask: pd.Series) -> pd.Series:
    y = series.astype(float)
    non_missing = ~missing_mask.astype(bool)
    positive = y[(y > 0) & non_missing]
    if positive.empty:
        return pd.Series(False, index=series.index)
    high_threshold = max(float(positive.quantile(0.995)), float(positive.median() + 8 * robust_scale(positive)))
    prev_low = y.shift(1).fillna(0) <= positive.quantile(0.25)
    next_low = y.shift(-1).fillna(0) <= positive.quantile(0.25)
    isolated_high = (y >= high_threshold) & prev_low & next_low & non_missing
    return isolated_high.fillna(False)


def count_anomaly(series: pd.Series, missing_mask: pd.Series) -> pd.Series:
    y = series.astype(float)
    non_missing = ~missing_mask.astype(bool)
    invalid = ((y < 0) | (np.abs(y - np.round(y)) > 1e-9)) & non_missing
    valid = y[non_missing & ~invalid]
    if valid.empty:
        return invalid.fillna(False)
    high_threshold = max(float(valid.quantile(0.995)), float(valid.median() + 6 * robust_scale(valid)))
    local = rolling_median(y, window=ROLLING_MEDIAN_WINDOW)
    residual = y - local
    local_high = residual > max(4.0 * robust_scale(residual[non_missing]), 2.0)
    return (invalid | ((y >= high_threshold) & local_high & non_missing)).fillna(False)


def continuous_anomaly(series: pd.Series, missing_mask: pd.Series) -> pd.Series:
    y = series.astype(float)
    non_missing = ~missing_mask.astype(bool)
    med = rolling_median(y, window=ROLLING_MEDIAN_WINDOW)
    residual = y - med
    local_mad = (
        residual.abs()
        .rolling(window=ANOMALY_WINDOW, center=True, min_periods=5)
        .median()
        .fillna(residual.abs().median())
    )
    local_scale = 1.4826 * local_mad
    global_scale = robust_scale(residual[non_missing])
    residual_flag = residual.abs() > np.maximum(6.0 * local_scale, 4.5 * global_scale)

    diff = y.diff()
    diff_scale = robust_scale(diff[non_missing])
    jump_flag = diff.abs() > 6.0 * diff_scale
    reverse_flag = (diff * diff.shift(-1) < 0) & (diff.shift(-1).abs() > 0.5 * diff.abs())
    return (non_missing & (residual_flag | (jump_flag & reverse_flag))).fillna(False)


def detect_anomalies(train: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame({"编号": train["编号"]})
    flags["a异常"] = rain_anomaly(train["a_补齐值"], train["a_原始缺失"])
    flags["b异常"] = continuous_anomaly(train["b_补齐值"], train["b_原始缺失"])
    flags["c异常"] = count_anomaly(train["c_补齐值"], train["c_原始缺失"])
    flags["d异常"] = continuous_anomaly(train["d_补齐值"], train["d_原始缺失"])
    flags["e异常"] = continuous_anomaly(train["e_补齐值"], train["e_原始缺失"])
    candidate_map = {
        "b": "突变候选标记_b_孔隙水压力_kPa",
        "d": "突变候选标记_d_深部位移_mm",
        "e": "突变候选标记_e_表面位移_mm",
    }
    for key, col in candidate_map.items():
        if col in train.columns:
            flags[f"{key}异常"] = flags[f"{key}异常"] | train[col].fillna(False).astype(bool)
    flag_cols = [f"{k}异常" for k in VAR_NAMES]
    flags["异常变量数量"] = flags[flag_cols].sum(axis=1).astype(int)
    flags["共同异常"] = flags["异常变量数量"] >= 2
    flags["异常变量编码"] = flags.apply(
        lambda row: "".join([k for k in VAR_NAMES if bool(row[f"{k}异常"])]), axis=1
    )
    return flags


def distribution_diagnostics(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    type_map = {
        "a": "事件型非负连续变量",
        "b": "连续监测变量",
        "c": "离散计数变量",
        "d": "连续监测变量",
        "e": "连续监测变量",
    }
    for key, raw_col in VAR_NAMES.items():
        model_col = f"{key}_建模值"
        missing_col = f"{key}_原始缺失"
        values = train[model_col].dropna().astype(float)
        raw_values = train.loc[~train[missing_col].astype(bool), raw_col].dropna().astype(float)
        mean_value = float(values.mean()) if len(values) else np.nan
        variance_value = float(values.var(ddof=1)) if len(values) > 1 else np.nan
        variance_mean_ratio = variance_value / mean_value if np.isfinite(mean_value) and mean_value > 0 else np.nan
        skew_value = float(values.skew()) if len(values) > 2 else np.nan
        kurt_value = float(values.kurt()) if len(values) > 3 else np.nan
        normal_like = (
            key in {"b", "d", "e"}
            and np.isfinite(skew_value)
            and np.isfinite(kurt_value)
            and abs(skew_value) <= 1.0
            and abs(kurt_value) <= 3.0
        )
        if key == "c":
            poisson_note = "方差均值比较仅作为离散性诊断，不作为主异常假设"
            criterion = "百分位数与局部突增"
        elif key == "a":
            poisson_note = "不适用"
            criterion = "高分位孤立峰值与非负约束"
        elif normal_like:
            poisson_note = "不适用"
            criterion = "3σ可作为对照，主判据仍保留MAD稳健规则"
        else:
            poisson_note = "不适用"
            criterion = "MAD稳健统计与局部趋势残差"
        rows.append(
            {
                "变量": f"{key}：{raw_col.split('_', 1)[1]}",
                "变量类型": type_map[key],
                "原始非缺失样本数": int(len(raw_values)),
                "原始缺失数": int(train[missing_col].sum()),
                "建模序列样本数": int(len(values)),
                "最小值": float(values.min()) if len(values) else np.nan,
                "最大值": float(values.max()) if len(values) else np.nan,
                "均值": mean_value,
                "中位数": float(values.median()) if len(values) else np.nan,
                "标准差": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "偏度": skew_value,
                "超额峰度": kurt_value,
                "零值比例": float((values == 0).mean()) if len(values) else np.nan,
                "P95": float(values.quantile(0.95)) if len(values) else np.nan,
                "P99": float(values.quantile(0.99)) if len(values) else np.nan,
                "P99.5": float(values.quantile(0.995)) if len(values) else np.nan,
                "方差均值比": variance_mean_ratio,
                "是否近似适用3σ": "是" if normal_like else "否",
                "建议异常判据": criterion,
                "泊松假设说明": poisson_note,
            }
        )
    return pd.DataFrame(rows)


def anomaly_rule_table(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, raw_col in VAR_NAMES.items():
        y = train[f"{key}_补齐值"].astype(float)
        non_missing = ~train[f"{key}_原始缺失"].astype(bool)
        valid = y[non_missing]
        if key == "a":
            positive = y[(y > 0) & non_missing]
            threshold = (
                max(float(positive.quantile(0.995)), float(positive.median() + 8 * robust_scale(positive)))
                if not positive.empty
                else np.nan
            )
            rows.append(
                {
                    "变量": f"{key}：{raw_col.split('_', 1)[1]}",
                    "变量属性": "降雨事件型变量",
                    "是否采用正态3σ": "否",
                    "主要阈值口径": "正降雨P99.5与中位数+8MAD取较大值，并要求前后邻点处于低雨量水平",
                    "滑动窗口或局部尺度": "不设主滑动窗口，仅使用邻点持续性约束",
                    "本轮主阈值": threshold,
                    "方法说明": "保留正常降雨峰值，仅将孤立且显著高于样本主体的峰值列为异常候选",
                }
            )
        elif key == "c":
            invalid = ((y < 0) | (np.abs(y - np.round(y)) > 1e-9)) & non_missing
            valid_count = y[non_missing & ~invalid]
            threshold = (
                max(float(valid_count.quantile(0.995)), float(valid_count.median() + 6 * robust_scale(valid_count)))
                if not valid_count.empty
                else np.nan
            )
            mean_value = float(valid_count.mean()) if len(valid_count) else np.nan
            var_value = float(valid_count.var(ddof=1)) if len(valid_count) > 1 else np.nan
            rows.append(
                {
                    "变量": f"{key}：{raw_col.split('_', 1)[1]}",
                    "变量属性": "非负整数计数变量",
                    "是否采用正态3σ": "否",
                    "主要阈值口径": "非负整数约束、P99.5与中位数+6MAD取较大值，并结合局部突增判据",
                    "滑动窗口或局部尺度": f"局部中位数窗口{ROLLING_MEDIAN_WINDOW}点",
                    "本轮主阈值": threshold,
                    "方法说明": f"泊松假设仅作诊断参考；样本均值={mean_value:.6g}，样本方差={var_value:.6g}",
                }
            )
        else:
            diff_scale = robust_scale(y.diff()[non_missing])
            residual = y - rolling_median(y, window=ROLLING_MEDIAN_WINDOW)
            global_residual_scale = robust_scale(residual[non_missing])
            rows.append(
                {
                    "变量": f"{key}：{raw_col.split('_', 1)[1]}",
                    "变量属性": "连续监测变量",
                    "是否采用正态3σ": "仅作为对照，不作为主规则",
                    "主要阈值口径": "局部趋势残差超过max(6局部MAD,4.5全局MAD)，或差分超过6全局MAD且随后反向回落",
                    "滑动窗口或局部尺度": f"趋势窗口{ROLLING_MEDIAN_WINDOW}点，局部MAD窗口{ANOMALY_WINDOW}点",
                    "本轮主阈值": float(4.5 * global_residual_scale),
                    "方法说明": f"差分辅助阈值为{6.0 * diff_scale:.6g}；同时继承第一轮保守预处理中已标记的突变候选点",
                }
            )
    return pd.DataFrame(rows)


def add_lag_features(df: pd.DataFrame, selected_lags: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    for key, lag in selected_lags.items():
        col = f"{key}_建模值"
        new_col = f"{key}_滞后{lag}步"
        out[new_col] = out[col].shift(lag).ffill().bfill()
    return out


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str | int]:
    return {
        "模型": name,
        "样本数": int(len(y_true)),
        "RMSE_mm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_mm": float(mean_absolute_error(y_true, y_pred)),
        "Bias_mm": float(np.mean(y_pred - y_true)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def blocked_cv(df: pd.DataFrame, feature_cols: list[str], target_col: str, model) -> tuple[np.ndarray, pd.DataFrame]:
    n = len(df)
    fold_ids = np.array_split(np.arange(n), CV_SPLITS)
    pred = np.full(n, np.nan, dtype=float)
    rows = []
    for fold, valid_idx in enumerate(fold_ids, start=1):
        train_idx = np.setdiff1d(np.arange(n), valid_idx)
        fitted = clone(model)
        fitted.fit(df.iloc[train_idx][feature_cols], df.iloc[train_idx][target_col])
        fold_pred = fitted.predict(df.iloc[valid_idx][feature_cols])
        pred[valid_idx] = fold_pred
        rows.append(
            {
                "折号": fold,
                "验证起始编号": int(df.iloc[valid_idx]["编号"].min()),
                "验证终止编号": int(df.iloc[valid_idx]["编号"].max()),
                **metric_row(
                    f"fold_{fold}",
                    df.iloc[valid_idx][target_col].to_numpy(dtype=float),
                    fold_pred,
                ),
            }
        )
    return pred, pd.DataFrame(rows)


def select_lags(train: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    rows = []
    selected: dict[str, int] = {}
    y = train["e_建模值"]
    base_corr = {}
    for key in ["a", "b"]:
        base_corr[key] = abs(train[f"{key}_建模值"].corr(y, method="spearman"))
        best_lag = None
        best_corr = -np.inf
        for lag in LAG_STEPS:
            lagged = train[f"{key}_建模值"].shift(lag)
            corr = abs(lagged.corr(y, method="spearman"))
            rows.append(
                {
                    "变量": key,
                    "滞后步数": lag,
                    "滞后时间_min": lag * 10,
                    "Spearman_abs": float(corr) if np.isfinite(corr) else np.nan,
                    "同编号Spearman_abs": float(base_corr[key]) if np.isfinite(base_corr[key]) else np.nan,
                }
            )
            if np.isfinite(corr) and corr > best_corr:
                best_corr = corr
                best_lag = lag
        if best_lag is not None and best_corr >= base_corr[key] + 0.03:
            selected[key] = int(best_lag)
    return selected, pd.DataFrame(rows)


def build_model_data(train: pd.DataFrame, flags: pd.DataFrame, selected_lags: dict[str, int]) -> tuple[pd.DataFrame, list[str]]:
    model_df = add_lag_features(train, selected_lags)
    feature_cols = [f"{k}_建模值" for k in FEATURE_KEYS]
    feature_cols += [f"{k}_滞后{lag}步" for k, lag in selected_lags.items()]

    valid = ~train["e_原始缺失"].astype(bool)
    model_df = model_df.loc[valid].copy()
    model_df["共同异常"] = flags.loc[valid, "共同异常"].to_numpy(dtype=bool)
    return model_df, feature_cols


def fit_and_compare(model_df: pd.DataFrame, feature_cols: list[str]) -> tuple[object, str, pd.DataFrame, pd.DataFrame, np.ndarray]:
    models = {
        "普通线性回归": make_pipeline(StandardScaler(), LinearRegression()),
        "岭回归": make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS)),
        "Huber稳健回归": make_pipeline(StandardScaler(), HuberRegressor(alpha=0.001, max_iter=1000)),
        "随机森林辅助对照": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=8,
            random_state=51,
            n_jobs=-1,
        ),
    }
    rows = []
    cv_tables = []
    cv_predictions = {}
    for name, model in models.items():
        pred, cv = blocked_cv(model_df, feature_cols, "e_建模值", model)
        cv.insert(0, "候选模型", name)
        cv_tables.append(cv)
        rows.append(metric_row(name, model_df["e_建模值"].to_numpy(dtype=float), pred))
        cv_predictions[name] = pred
    comparison = pd.DataFrame(rows)
    cv_table = pd.concat(cv_tables, ignore_index=True)

    ridge_rmse = float(comparison.loc[comparison["模型"] == "岭回归", "RMSE_mm"].iloc[0])
    rf_rmse = float(comparison.loc[comparison["模型"] == "随机森林辅助对照", "RMSE_mm"].iloc[0])
    main_name = "岭回归"
    if rf_rmse < 0.90 * ridge_rmse:
        main_name = "随机森林辅助对照"
    main_model = clone(models[main_name])
    main_model.fit(model_df[feature_cols], model_df["e_建模值"])
    return main_model, main_name, comparison, cv_table, cv_predictions[main_name]


def contribution_table(model, model_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = model_df[feature_cols]
    y = model_df["e_建模值"]
    fitted_pred = model.predict(X)
    base_rmse = float(np.sqrt(mean_squared_error(y, fitted_pred)))

    coef_values: dict[str, float] = {}
    if hasattr(model, "named_steps") and "ridgecv" in model.named_steps:
        coefs = model.named_steps["ridgecv"].coef_
        coef_values = {col: float(abs(coef)) for col, coef in zip(feature_cols, coefs)}
    elif hasattr(model, "named_steps") and "huberregressor" in model.named_steps:
        coefs = model.named_steps["huberregressor"].coef_
        coef_values = {col: float(abs(coef)) for col, coef in zip(feature_cols, coefs)}
    elif hasattr(model, "feature_importances_"):
        coef_values = {col: float(val) for col, val in zip(feature_cols, model.feature_importances_)}
    else:
        coef_values = {col: np.nan for col in feature_cols}

    perm = permutation_importance(model, X, y, n_repeats=20, random_state=51, scoring="neg_root_mean_squared_error")
    rows = []
    for idx, col in enumerate(feature_cols):
        X_removed = X.copy()
        X_removed[col] = X_removed[col].median()
        pred_removed = model.predict(X_removed)
        rmse_removed = float(np.sqrt(mean_squared_error(y, pred_removed)))
        rows.append(
            {
                "变量": col,
                "标准化系数或重要性_绝对值": coef_values.get(col, np.nan),
                "置换重要性_RMSE增量": float(max(0.0, perm.importances_mean[idx])),
                "变量剔除_RMSE增量": float(rmse_removed - base_rmse),
            }
        )
    result = pd.DataFrame(rows)
    for col in ["标准化系数或重要性_绝对值", "置换重要性_RMSE增量", "变量剔除_RMSE增量"]:
        denom = result[col].sum()
        result[f"{col}_归一化贡献"] = result[col] / denom if denom > 0 else np.nan
    return result


def set_chinese_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans SC",
        "Source Han Sans CN",
        "DejaVu Sans",
    ]


def make_figures(paths: Paths, train: pd.DataFrame, test: pd.DataFrame, flags: pd.DataFrame, pred_test: pd.DataFrame) -> None:
    set_chinese_font()

    miss_counts = train[[f"{k}_原始缺失" for k in VAR_NAMES]].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(list(VAR_NAMES.keys()), miss_counts.values)
    ax.set_xlabel("变量")
    ax.set_ylabel("缺失记录数")
    ax.set_title("问题3训练集缺失记录统计")
    fig.tight_layout()
    fig.savefig(paths.figures / "问题3_训练集缺失统计.png", dpi=180)
    plt.close(fig)

    for key, col in VAR_NAMES.items():
        if key == "e" and "e_建模值" not in train:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(train["编号"], train[col], color="#9aa0a6", linewidth=0.8, label="原始值")
        ax.plot(train["编号"], train[f"{key}_建模值"], color="#1f77b4", linewidth=1.0, label="建模值")
        anomalous = flags[f"{key}异常"]
        if anomalous.any():
            ax.scatter(train.loc[anomalous, "编号"], train.loc[anomalous, f"{key}_补齐值"], s=12, color="#d62728", label="异常标记")
        ax.set_xlabel("编号")
        ax.set_ylabel(col)
        ax.set_title(f"问题3训练集{key}变量原始值与建模值")
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths.figures / f"问题3_{key}_原始与建模序列.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, key in zip(axes.ravel(), FEATURE_KEYS):
        ax.scatter(train[f"{key}_建模值"], train["e_建模值"], s=8, alpha=0.35)
        ax.set_xlabel(f"{key}_建模值")
        ax.set_ylabel("e_建模值")
        ax.set_title(f"{key}-e散点关系")
    fig.tight_layout()
    fig.savefig(paths.figures / "问题3_解释变量与表面位移散点图.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(pred_test["编号"], pred_test["表面位移估计值_mm"], color="#1f77b4", linewidth=1.0)
    ax.set_xlabel("编号")
    ax.set_ylabel("表面位移估计值/mm")
    ax.set_title("问题3实验集表面位移估计序列")
    fig.tight_layout()
    fig.savefig(paths.figures / "问题3_实验集表面位移估计序列.png", dpi=180)
    plt.close(fig)


def make_distribution_figures(paths: Paths, train: pd.DataFrame) -> None:
    set_chinese_font()
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    axes_flat = axes.ravel()
    for ax, (key, raw_col) in zip(axes_flat, VAR_NAMES.items()):
        values = train[f"{key}_建模值"].dropna().astype(float)
        bins = min(60, max(12, int(np.sqrt(len(values)))))
        ax.hist(values, bins=bins, color="#4c78a8", alpha=0.78, edgecolor="white")
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=1))
        median_value = float(values.median())
        mad_scale = robust_scale(values)
        ax.axvline(mean_value, color="#d62728", linewidth=1.1, label="均值")
        ax.axvline(median_value, color="#2ca02c", linewidth=1.1, label="中位数")
        if np.isfinite(std_value) and std_value > 0:
            ax.axvline(mean_value + 3 * std_value, color="#d62728", linewidth=0.8, linestyle="--")
            ax.axvline(mean_value - 3 * std_value, color="#d62728", linewidth=0.8, linestyle="--")
        if np.isfinite(mad_scale) and mad_scale > 0:
            ax.axvline(median_value + 3 * mad_scale, color="#2ca02c", linewidth=0.8, linestyle=":")
            ax.axvline(median_value - 3 * mad_scale, color="#2ca02c", linewidth=0.8, linestyle=":")
        ax.set_title(f"{key}：{raw_col.split('_', 1)[1]}分布")
        ax.set_xlabel("建模序列取值")
        ax.set_ylabel("频数")
        ax.legend(fontsize=8)
    axes_flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(paths.figures / "问题3_变量分布直方图.png", dpi=180)
    plt.close(fig)


def make_residual_figures(paths: Paths, model_df: pd.DataFrame) -> None:
    set_chinese_font()
    y_true = model_df["e_建模值"].to_numpy(dtype=float)
    y_pred = model_df["交叉验证预测值_mm"].to_numpy(dtype=float)
    residual = y_pred - y_true

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(y_true, y_pred, s=10, alpha=0.35, color="#1f77b4")
    lower = float(min(np.min(y_true), np.min(y_pred)))
    upper = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lower, upper], [lower, upper], color="#d62728", linewidth=1.2, label="理想预测线")
    ax.set_xlabel("表面位移真实值/mm")
    ax.set_ylabel("表面位移交叉验证预测值/mm")
    ax.set_title("问题3表面位移真实值与预测值散点图")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figures / "问题3_真实值预测值散点图.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.scatter(model_df["编号"], residual, s=8, alpha=0.35, color="#1f77b4")
    ax.set_xlabel("编号")
    ax.set_ylabel("交叉验证残差/mm")
    ax.set_title("问题3残差随编号变化图")
    fig.tight_layout()
    fig.savefig(paths.figures / "问题3_残差随编号变化图.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(residual, bins=45, color="#1f77b4", alpha=0.78, edgecolor="white")
    ax.axvline(0.0, color="#d62728", linewidth=1.2)
    ax.set_xlabel("交叉验证残差/mm")
    ax.set_ylabel("频数")
    ax.set_title("问题3残差分布图")
    fig.tight_layout()
    fig.savefig(paths.figures / "问题3_残差分布图.png", dpi=180)
    plt.close(fig)


def write_outputs(
    paths: Paths,
    train_processed: pd.DataFrame,
    test_processed: pd.DataFrame,
    flags: pd.DataFrame,
    table31: pd.DataFrame,
    table32: pd.DataFrame,
    correlation: pd.DataFrame,
    lag_table: pd.DataFrame,
    model_comparison: pd.DataFrame,
    cv_table: pd.DataFrame,
    contribution: pd.DataFrame,
    distribution_stats: pd.DataFrame,
    anomaly_rules: pd.DataFrame,
    pred_test: pd.DataFrame,
    model_df: pd.DataFrame,
    log: dict,
) -> None:
    with pd.ExcelWriter(paths.data / "问题3_训练集预处理结果.xlsx") as writer:
        train_processed.to_excel(writer, sheet_name="训练集预处理", index=False)
        flags.to_excel(writer, sheet_name="异常标记", index=False)
        distribution_stats.to_excel(writer, sheet_name="分布统计", index=False)
        anomaly_rules.to_excel(writer, sheet_name="异常判据", index=False)
    with pd.ExcelWriter(paths.data / "问题3_实验集特征预处理结果.xlsx") as writer:
        test_processed.to_excel(writer, sheet_name="实验集特征预处理", index=False)
        pred_test.to_excel(writer, sheet_name="表面位移估计", index=False)

    train_processed.to_csv(paths.data / "问题3_训练集预处理结果.csv", index=False, encoding="utf-8-sig")
    test_processed.to_csv(paths.data / "问题3_实验集特征预处理结果.csv", index=False, encoding="utf-8-sig")
    pred_test.to_csv(paths.tables / "问题3_实验集表面位移估计.csv", index=False, encoding="utf-8-sig")
    model_df[
        ["编号", "e_建模值", "交叉验证预测值_mm", "共同异常"]
    ].to_csv(paths.tables / "问题3_交叉验证预测明细.csv", index=False, encoding="utf-8-sig")

    table31.to_csv(paths.tables / "表3_1_单变量异常点数量.csv", index=False, encoding="utf-8-sig")
    table32.to_csv(paths.tables / "表3_2_共同异常点清单.csv", index=False, encoding="utf-8-sig")
    correlation.to_csv(paths.tables / "问题3_关联分析指标.csv", index=False, encoding="utf-8-sig")
    lag_table.to_csv(paths.tables / "问题3_滞后相关性观察.csv", index=False, encoding="utf-8-sig")
    model_comparison.to_csv(paths.tables / "问题3_模型检验指标.csv", index=False, encoding="utf-8-sig")
    cv_table.to_csv(paths.tables / "问题3_时间分块交叉验证.csv", index=False, encoding="utf-8-sig")
    contribution.to_csv(paths.tables / "问题3_变量贡献度.csv", index=False, encoding="utf-8-sig")
    distribution_stats.to_csv(paths.tables / "问题3_变量分布统计.csv", index=False, encoding="utf-8-sig")
    anomaly_rules.to_csv(paths.tables / "问题3_异常判据说明.csv", index=False, encoding="utf-8-sig")

    with open(paths.logs / "问题3_处理日志.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    with open(paths.logs / "问题3_处理日志.txt", "w", encoding="utf-8") as f:
        for key, value in log.items():
            f.write(f"{key}: {value}\n")


def main() -> None:
    root = Path(__file__).resolve().parent
    paths = make_paths(root)
    source_path = locate_preprocessed_attachment3(root)
    train_raw = pd.read_excel(source_path, sheet_name="训练集")
    test_raw = pd.read_excel(source_path, sheet_name="实验集")

    train_processed = prepare_one_frame(train_raw, is_train=True)
    test_processed = prepare_one_frame(test_raw, is_train=False)

    flags = detect_anomalies(train_processed)
    for col in [c for c in flags.columns if c.endswith("异常") or c in ["共同异常", "异常变量数量", "异常变量编码"]]:
        train_processed[col] = flags[col]

    distribution_stats = distribution_diagnostics(train_processed)
    anomaly_rules = anomaly_rule_table(train_processed)

    table31_rows = []
    for key, name in VAR_NAMES.items():
        table31_rows.append(
            {
                "数据集变量": f"{key}：{name.split('_', 1)[1]}",
                "异常点数量": int(flags[f"{key}异常"].sum()),
            }
        )
    table31_rows.append({"数据集变量": "总数", "异常点数量": int(sum(row["异常点数量"] for row in table31_rows))})
    table31 = pd.DataFrame(table31_rows)

    table32 = flags.loc[flags["共同异常"], ["编号", "异常变量编码", "异常变量数量"]].rename(
        columns={"编号": "时间点对应编号", "异常变量编码": "共同异常点处的异常变量"}
    )

    selected_lags, lag_table = select_lags(train_processed.loc[~train_processed["e_原始缺失"]].copy())
    model_df, feature_cols = build_model_data(train_processed, flags, selected_lags)
    main_model, main_model_name, model_comparison, cv_table, cv_pred = fit_and_compare(model_df, feature_cols)
    model_df["交叉验证预测值_mm"] = cv_pred

    contribution = contribution_table(main_model, model_df, feature_cols)

    corr_rows = []
    for key in FEATURE_KEYS:
        corr_rows.append(
            {
                "解释变量": f"{key}_建模值",
                "Pearson相关系数": float(train_processed[f"{key}_建模值"].corr(train_processed["e_建模值"], method="pearson")),
                "Spearman相关系数": float(train_processed[f"{key}_建模值"].corr(train_processed["e_建模值"], method="spearman")),
            }
        )
    correlation = pd.DataFrame(corr_rows)

    test_for_pred = add_lag_features(test_processed, selected_lags)
    pred_values = main_model.predict(test_for_pred[feature_cols])
    pred_test = pd.DataFrame(
        {
            "编号": test_processed["编号"],
            "表面位移估计值_mm": pred_values,
        }
    )
    test_processed["e_表面位移估计值_mm"] = pred_values

    make_figures(paths, train_processed, test_processed, flags, pred_test)
    make_distribution_figures(paths, train_processed)
    make_residual_figures(paths, model_df)

    log = {
        "输入文件": str(source_path.relative_to(root)),
        "训练集样本数": int(len(train_processed)),
        "实验集样本数": int(len(test_processed)),
        "训练集缺失数": {key: int(train_processed[f"{key}_原始缺失"].sum()) for key in VAR_NAMES},
        "实验集解释变量缺失数": {key: int(test_processed[f"{key}_原始缺失"].sum()) for key in FEATURE_KEYS},
        "异常点数量": {key: int(flags[f"{key}异常"].sum()) for key in VAR_NAMES},
        "共同异常点数量": int(flags["共同异常"].sum()),
        "表3_1总数口径": "五个变量异常次数之和，不是异常编号去重数。",
        "异常编号去重数": int((flags["异常变量数量"] > 0).sum()),
        "分布诊断口径": "对五个变量的建模序列计算直方图、偏度、超额峰度和高分位数，用于判断3σ、MAD及百分位数规则的适用性。",
        "异常阈值口径": "连续监测变量以局部趋势残差和MAD稳健阈值为主；降雨采用高分位孤立峰值；微震事件数采用非负整数约束、百分位数阈值和局部突增判据。",
        "微震泊松口径": "微震事件数为离散计数变量，但边坡损伤过程可能存在聚集性和阶段性，泊松分布仅作为离散性诊断，不作为主异常检测假设。",
        "候选滞后步": LAG_STEPS,
        "纳入主模型的滞后项": selected_lags,
        "建模样本口径": "训练集表面位移原始观测非缺失记录；解释变量使用问题3.1补齐与温和去噪后的建模值。",
        "建模样本数": int(len(model_df)),
        "主模型": main_model_name,
        "特征列": feature_cols,
    }

    write_outputs(
        paths,
        train_processed,
        test_processed,
        flags,
        table31,
        table32,
        correlation,
        lag_table,
        model_comparison,
        cv_table,
        contribution,
        distribution_stats,
        anomaly_rules,
        pred_test,
        model_df,
        log,
    )

    print("问题3处理完成。")
    print(f"输出目录：{paths.out}")
    print(table31.to_string(index=False))
    print(model_comparison.to_string(index=False))
    print(f"共同异常点数量：{int(flags['共同异常'].sum())}")
    print(f"实验集预测数量：{len(pred_test)}")


if __name__ == "__main__":
    main()
