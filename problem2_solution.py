from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


START_TIME = "2024-05-04 00:00:00"
FREQ = "10min"
DT_HOURS = 1.0 / 6.0
MAIN_SMOOTH_WINDOW = 61
SENSITIVITY_WINDOWS = [31, 61, 121]
MIN_SEGMENT_LENGTH = 800


@dataclass(frozen=True)
class Paths:
    root: Path
    out: Path
    data: Path
    tables: Path
    figures: Path
    logs: Path


def make_paths(root: Path) -> Paths:
    out = root / "problem2_outputs"
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


def locate_attachment2(root: Path) -> Path:
    candidates = [
        p
        for p in root.rglob("*.xlsx")
        if "preprocess_outputs" not in str(p)
        and "problem1_outputs" not in str(p)
        and "problem2_outputs" not in str(p)
        and ("附件2" in p.name or "问题2" in p.name)
    ]
    if not candidates:
        originals = sorted(
            [
                p
                for p in root.rglob("*.xlsx")
                if "preprocess_outputs" not in str(p)
                and "problem1_outputs" not in str(p)
                and "problem2_outputs" not in str(p)
            ],
            key=lambda p: p.name,
        )
        if len(originals) < 2:
            raise FileNotFoundError("未找到附件2 Excel 文件。")
        return originals[1]
    return sorted(candidates, key=lambda p: (len(str(p)), p.name))[0]


def prepare_trend(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    y_for_trend = y.copy()
    non_initial_zero = (y_for_trend == 0) & (np.arange(len(y_for_trend)) > 0)
    series = pd.Series(y_for_trend)
    series.loc[non_initial_zero] = np.nan
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
        if den == 0:
            return float(sum_yy - sum_y * sum_y / m)
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


def best_one_break(y: np.ndarray, min_len: int = MIN_SEGMENT_LENGTH, step: int = 10):
    n = len(y)
    sse = segment_sse_function(y)
    best = (float("inf"), None)
    for t in range(min_len, n - min_len, step):
        value = sse(0, t) + sse(t, n)
        if value < best[0]:
            best = (value, t)
    center = best[1]
    if center is None:
        raise RuntimeError("无法搜索两段分段点。")
    left = max(min_len, center - 50)
    right = min(n - min_len, center + 51)
    for t in range(left, right):
        value = sse(0, t) + sse(t, n)
        if value < best[0]:
            best = (value, t)
    return best


def best_two_breaks(y: np.ndarray, min_len: int = MIN_SEGMENT_LENGTH):
    n = len(y)
    sse = segment_sse_function(y)
    best = (float("inf"), None, None)
    for t1 in range(min_len, n - 2 * min_len, 20):
        for t2 in range(t1 + min_len, n - min_len, 20):
            value = sse(0, t1) + sse(t1, t2) + sse(t2, n)
            if value < best[0]:
                best = (value, t1, t2)
    if best[1] is None or best[2] is None:
        raise RuntimeError("无法搜索三段分段点。")

    _, center1, center2 = best
    fine_best = best
    left1 = max(min_len, center1 - 100)
    right1 = min(n - 2 * min_len, center1 + 101)
    left2 = max(center1 + min_len - 100, center2 - 100)
    right2 = min(n - min_len, center2 + 101)
    for t1 in range(left1, right1):
        start2 = max(t1 + min_len, left2)
        for t2 in range(start2, right2):
            if n - t2 < min_len:
                continue
            value = sse(0, t1) + sse(t1, t2) + sse(t2, n)
            if value < fine_best[0]:
                fine_best = (value, t1, t2)
    return fine_best


def model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE_mm": rmse, "MAE_mm": mae, "R2": r2}


def fit_poly(t_hours: np.ndarray, y: np.ndarray, degree: int):
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
    model.fit(t_hours.reshape(-1, 1), y)
    pred = model.predict(t_hours.reshape(-1, 1))
    reg = model.named_steps["linearregression"]
    coefficients = reg.coef_
    intercept = float(reg.intercept_)
    return model, pred, intercept, coefficients


def stage_degree(stage_no: int, t_hours: np.ndarray, trend: np.ndarray) -> int:
    if stage_no == 1:
        return 1
    if stage_no == 2:
        return 2
    _, pred1, _, _ = fit_poly(t_hours, trend, 1)
    _, pred2, _, _ = fit_poly(t_hours, trend, 2)
    rmse1 = np.sqrt(mean_squared_error(trend, pred1))
    rmse2 = np.sqrt(mean_squared_error(trend, pred2))
    return 2 if rmse2 <= 0.95 * rmse1 else 1


def robust_jump_flags(y: np.ndarray) -> tuple[np.ndarray, pd.Series]:
    increments = pd.Series(np.diff(y))
    local_median = increments.rolling(61, center=True, min_periods=20).median()
    local_mad = (increments - local_median).abs().rolling(61, center=True, min_periods=20).median()
    score = ((increments - local_median).abs() / (1.4826 * local_mad.replace(0, np.nan))).replace(
        [np.inf, -np.inf], np.nan
    )
    flags = np.zeros(len(y), dtype=bool)
    idx = np.where(score > 6)[0] + 1
    flags[idx] = True
    return flags, score


def stage_name(stage_no: int) -> str:
    return {
        1: "缓慢匀速形变阶段",
        2: "加速形变阶段",
        3: "快速形变阶段",
    }[stage_no]


def save_figures(
    paths: Paths,
    result: pd.DataFrame,
    time_col: str,
    y_col: str,
    t1: int,
    t2: int,
    stage_models: dict[int, np.ndarray],
) -> None:
    times = pd.to_datetime(result[time_col])
    raw = result[y_col].to_numpy(dtype=float)
    trend = result["稳健趋势位移_mm"].to_numpy(dtype=float)
    velocity = result["滚动72点速度_mm_h"].to_numpy(dtype=float)
    acceleration = result["滚动72点加速度_mm_h2"].to_numpy(dtype=float)

    plt.figure(figsize=(10.0, 5.2))
    plt.plot(times, raw, linewidth=0.7, alpha=0.45, label="Original displacement")
    plt.plot(times, trend, linewidth=1.4, color="#c43c39", label="Robust trend")
    for idx, label in [(t1, "Transition 1"), (t2, "Transition 2")]:
        plt.axvline(times.iloc[idx], color="#222222", linestyle="--", linewidth=1.0)
        plt.text(times.iloc[idx], np.nanmax(raw) * 0.92, label, rotation=90, va="top")
    plt.xlabel("Time")
    plt.ylabel("Surface displacement (mm)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(paths.figures / "问题2_位移阶段划分图.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10.0, 5.8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(times, velocity, linewidth=0.8, color="#2f6f9f")
    ax1.axvline(times.iloc[t1], color="#222222", linestyle="--", linewidth=1.0)
    ax1.axvline(times.iloc[t2], color="#222222", linestyle="--", linewidth=1.0)
    ax1.set_ylabel("Velocity (mm/h)")
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(times, acceleration, linewidth=0.8, color="#7b6d3a")
    ax2.axvline(times.iloc[t1], color="#222222", linestyle="--", linewidth=1.0)
    ax2.axvline(times.iloc[t2], color="#222222", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Acceleration (mm/h^2)")
    ax2.set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(paths.figures / "问题2_速度加速度诊断图.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10.0, 5.2))
    plt.plot(times, raw, linewidth=0.65, alpha=0.35, label="Original displacement")
    colors = {1: "#1b8a5a", 2: "#c47f2c", 3: "#b33d3d"}
    for stage_no, pred in stage_models.items():
        mask = result["阶段编号"].to_numpy() == stage_no
        plt.plot(
            times[mask],
            pred,
            linewidth=1.8,
            color=colors[stage_no],
            label=f"Stage {stage_no} model",
        )
    plt.axvline(times.iloc[t1], color="#222222", linestyle="--", linewidth=1.0)
    plt.axvline(times.iloc[t2], color="#222222", linestyle="--", linewidth=1.0)
    plt.xlabel("Time")
    plt.ylabel("Surface displacement (mm)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(paths.figures / "问题2_分阶段拟合图.png", dpi=200)
    plt.close()


def main() -> None:
    root = Path.cwd()
    paths = make_paths(root)
    source = locate_attachment2(root)
    df = pd.read_excel(source)

    id_col = df.columns[0]
    y_col = df.columns[1]
    y = df[y_col].astype(float).to_numpy()
    n = len(df)
    expected_ids = np.arange(1, n + 1)
    id_continuous = bool(np.array_equal(df[id_col].to_numpy(), expected_ids))
    times = pd.date_range(START_TIME, periods=n, freq=FREQ)

    non_initial_zero = (y == 0) & (np.arange(n) > 0)
    interpolated_for_trend, trend = prepare_trend(y, MAIN_SMOOTH_WINDOW)
    jump_flags, jump_score = robust_jump_flags(y)

    _, one_break = best_one_break(trend)
    two_sse, t1, t2 = best_two_breaks(trend)
    sse = segment_sse_function(trend)
    single_sse = sse(0, n)
    one_sse = sse(0, one_break) + sse(one_break, n)

    sensitivity_rows: list[dict[str, object]] = []
    for window in SENSITIVITY_WINDOWS:
        _, trend_w = prepare_trend(y, window)
        value, t1_w, t2_w = best_two_breaks(trend_w)
        sensitivity_rows.append(
            {
                "平滑窗口": window,
                "第一转换点序号": int(t1_w + 1),
                "第一转换点时间": str(times[t1_w]),
                "第二转换点序号": int(t2_w + 1),
                "第二转换点时间": str(times[t2_w]),
                "三段线性SSE": value,
            }
        )
    sensitivity_table = pd.DataFrame(sensitivity_rows)

    result = df.copy()
    result.insert(1, "推算时间", times)
    result["非初始零值标记"] = non_initial_zero
    result["局部差分突变候选标记"] = jump_flags
    score_full = np.full(n, np.nan)
    score_full[1:] = jump_score.to_numpy(dtype=float)
    result["局部差分突变评分"] = score_full
    result["趋势识别用位移_mm"] = interpolated_for_trend
    result["稳健趋势位移_mm"] = trend
    result["原始速度_mm_h"] = pd.Series(y).diff() / DT_HOURS
    result["滚动72点速度_mm_h"] = pd.Series(interpolated_for_trend).diff(72) / (72 * DT_HOURS)
    result["滚动72点加速度_mm_h2"] = result["滚动72点速度_mm_h"].diff(72) / (72 * DT_HOURS)

    stage = np.ones(n, dtype=int)
    stage[t1:t2] = 2
    stage[t2:] = 3
    result["阶段编号"] = stage
    result["阶段名称"] = [stage_name(int(s)) for s in stage]

    transition_table = pd.DataFrame(
        [
            {
                "转换类型": "缓慢匀速形变阶段->加速形变阶段",
                "转换点序号": int(t1 + 1),
                "转换点时间": str(times[t1]),
                "转换点原始位移_mm": float(y[t1]),
                "转换点稳健趋势位移_mm": float(trend[t1]),
                "前一阶段终止序号": int(t1),
                "后一阶段起始序号": int(t1 + 1),
            },
            {
                "转换类型": "加速形变阶段->快速形变阶段",
                "转换点序号": int(t2 + 1),
                "转换点时间": str(times[t2]),
                "转换点原始位移_mm": float(y[t2]),
                "转换点稳健趋势位移_mm": float(trend[t2]),
                "前一阶段终止序号": int(t2),
                "后一阶段起始序号": int(t2 + 1),
            },
        ]
    )

    model_rows: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []
    speed_rows: list[dict[str, object]] = []
    stage_model_predictions: dict[int, np.ndarray] = {}
    model_fit_full = np.full(n, np.nan)

    segments = [(1, 0, t1), (2, t1, t2), (3, t2, n)]
    for stage_no, left, right in segments:
        local_t = np.arange(right - left, dtype=float) * DT_HOURS
        y_trend = trend[left:right]
        y_raw = y[left:right]
        degree = stage_degree(stage_no, local_t, y_trend)
        model, pred_trend, intercept, coefficients = fit_poly(local_t, y_trend, degree)
        stage_model_predictions[stage_no] = pred_trend
        model_fit_full[left:right] = pred_trend
        metrics_trend = model_metrics(y_trend, pred_trend)
        metrics_raw = model_metrics(y_raw, pred_trend)

        expression = f"y={intercept:.6f}"
        for i, coef in enumerate(coefficients, start=1):
            expression += f"+({coef:.9f})*t^{i}"
        model_rows.append(
            {
                "阶段编号": stage_no,
                "阶段名称": stage_name(stage_no),
                "样本起始序号": int(left + 1),
                "样本终止序号": int(right),
                "起始时间": str(times[left]),
                "终止时间": str(times[right - 1]),
                "模型类型": f"{degree}次多项式趋势模型",
                "时间变量说明": "t为阶段内相对时间，单位h",
                "截距": intercept,
                "一次项系数_mm_h": float(coefficients[0]) if len(coefficients) >= 1 else np.nan,
                "二次项系数_mm_h2": float(coefficients[1]) if len(coefficients) >= 2 else np.nan,
                "模型表达式": expression,
            }
        )
        metrics_rows.append(
            {
                "阶段编号": stage_no,
                "阶段名称": stage_name(stage_no),
                "拟合对象": "稳健趋势位移",
                **metrics_trend,
            }
        )
        metrics_rows.append(
            {
                "阶段编号": stage_no,
                "阶段名称": stage_name(stage_no),
                "拟合对象": "原始位移相对趋势模型",
                **metrics_raw,
            }
        )
        duration_h = (right - left - 1) * DT_HOURS
        displacement_change = float(y[right - 1] - y[left])
        speed_rows.append(
            {
                "阶段编号": stage_no,
                "阶段名称": stage_name(stage_no),
                "起始序号": int(left + 1),
                "终止序号": int(right),
                "起始时间": str(times[left]),
                "终止时间": str(times[right - 1]),
                "起始原始位移_mm": float(y[left]),
                "终止原始位移_mm": float(y[right - 1]),
                "位移变化量_mm": displacement_change,
                "持续时间_h": duration_h,
                "阶段平均速度_mm_h": displacement_change / duration_h,
                "滚动72点速度中位数_mm_h": float(result.loc[left:right - 1, "滚动72点速度_mm_h"].median()),
            }
        )

    result["分阶段模型拟合位移_mm"] = model_fit_full
    result["分阶段模型残差_mm"] = result[y_col] - result["分阶段模型拟合位移_mm"]

    model_table = pd.DataFrame(model_rows)
    metrics_table = pd.DataFrame(metrics_rows)
    speed_table = pd.DataFrame(speed_rows)
    comparison_table = pd.DataFrame(
        [
            {"模型": "单段线性趋势", "断点数量": 0, "SSE": single_sse},
            {"模型": "两段线性趋势", "断点数量": 1, "SSE": one_sse},
            {"模型": "三段线性趋势", "断点数量": 2, "SSE": two_sse},
        ]
    )
    comparison_table["相对单段SSE下降率"] = 1 - comparison_table["SSE"] / single_sse

    zero_log = result.loc[
        result["非初始零值标记"],
        [id_col, "推算时间", y_col, "非初始零值标记"],
    ].copy()
    jump_log = result.loc[
        result["局部差分突变候选标记"],
        [id_col, "推算时间", y_col, "局部差分突变评分"],
    ].copy()

    save_figures(paths, result, "推算时间", y_col, t1, t2, stage_model_predictions)

    excel_path = paths.data / "问题2_阶段划分结果.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="阶段划分明细", index=False)
        transition_table.to_excel(writer, sheet_name="阶段转换节点", index=False)
        model_table.to_excel(writer, sheet_name="分阶段模型参数", index=False)
        speed_table.to_excel(writer, sheet_name="阶段平均速度", index=False)
        metrics_table.to_excel(writer, sheet_name="模型检验指标", index=False)
        comparison_table.to_excel(writer, sheet_name="分段数对照", index=False)
        sensitivity_table.to_excel(writer, sheet_name="平滑窗口敏感性", index=False)
        zero_log.to_excel(writer, sheet_name="非初始零值日志", index=False)
        jump_log.to_excel(writer, sheet_name="突变候选日志", index=False)

    result.to_csv(paths.data / "问题2_阶段划分明细.csv", index=False, encoding="utf-8-sig")
    transition_table.to_csv(paths.tables / "问题2_阶段转换节点.csv", index=False, encoding="utf-8-sig")
    model_table.to_csv(paths.tables / "问题2_分阶段模型参数.csv", index=False, encoding="utf-8-sig")
    speed_table.to_csv(paths.tables / "问题2_阶段平均速度.csv", index=False, encoding="utf-8-sig")
    metrics_table.to_csv(paths.tables / "问题2_模型检验指标.csv", index=False, encoding="utf-8-sig")
    comparison_table.to_csv(paths.tables / "问题2_分段数对照.csv", index=False, encoding="utf-8-sig")
    sensitivity_table.to_csv(paths.tables / "问题2_平滑窗口敏感性.csv", index=False, encoding="utf-8-sig")
    zero_log.to_csv(paths.logs / "问题2_非初始零值日志.csv", index=False, encoding="utf-8-sig")
    jump_log.to_csv(paths.logs / "问题2_局部差分突变候选日志.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source_file": str(source),
        "rows": int(n),
        "id_continuous": id_continuous,
        "time_start": str(times[0]),
        "time_end": str(times[-1]),
        "missing_counts": {str(k): int(v) for k, v in df.isna().sum().items()},
        "non_initial_zero_count": int(non_initial_zero.sum()),
        "jump_candidate_count": int(jump_flags.sum()),
        "main_smooth_window": MAIN_SMOOTH_WINDOW,
        "transition_1": {
            "index_0_based": int(t1),
            "sequence_no": int(t1 + 1),
            "time": str(times[t1]),
            "raw_displacement": float(y[t1]),
            "trend_displacement": float(trend[t1]),
        },
        "transition_2": {
            "index_0_based": int(t2),
            "sequence_no": int(t2 + 1),
            "time": str(times[t2]),
            "raw_displacement": float(y[t2]),
            "trend_displacement": float(trend[t2]),
        },
        "outputs": {
            "excel": str(excel_path),
            "transition_table": str(paths.tables / "问题2_阶段转换节点.csv"),
            "stage_models": str(paths.tables / "问题2_分阶段模型参数.csv"),
            "stage_speed": str(paths.tables / "问题2_阶段平均速度.csv"),
            "metrics": str(paths.tables / "问题2_模型检验指标.csv"),
            "figures": [str(p) for p in sorted(paths.figures.glob("问题2_*.png"))],
        },
    }
    with (paths.logs / "问题2_建模日志.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    text_log = [
        "问题2：三段式形变阶段识别日志",
        f"原始文件：{source}",
        f"样本量：{n}",
        f"编号连续：{id_continuous}",
        f"时间范围：{times[0]} 至 {times[-1]}",
        f"非初始零值记录数量：{int(non_initial_zero.sum())}",
        f"局部差分突变候选数量：{int(jump_flags.sum())}",
        f"主平滑窗口：{MAIN_SMOOTH_WINDOW} 个采样点",
        "",
        "阶段转换节点：",
        transition_table.to_string(index=False),
        "",
        "阶段平均速度：",
        speed_table.to_string(index=False),
        "",
        "分阶段模型参数：",
        model_table.to_string(index=False),
        "",
        "模型检验指标：",
        metrics_table.to_string(index=False),
        "",
        "分段数对照：",
        comparison_table.to_string(index=False),
        "",
        "平滑窗口敏感性：",
        sensitivity_table.to_string(index=False),
    ]
    (paths.logs / "问题2_建模日志.txt").write_text("\n".join(text_log), encoding="utf-8")

    print("Problem 2 stage identification completed.")
    print(f"Source: {source}")
    print(f"Output directory: {paths.out}")
    print(transition_table.to_string(index=False))
    print(speed_table[["阶段编号", "阶段名称", "阶段平均速度_mm_h"]].to_string(index=False))
    print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
