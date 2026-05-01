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
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


TARGET_VALUES = np.array([7.132, 18.526, 84.337, 123.554, 167.667], dtype=float)


@dataclass(frozen=True)
class Paths:
    root: Path
    out: Path
    data: Path
    tables: Path
    figures: Path
    logs: Path


def make_paths(root: Path) -> Paths:
    out = root / "problem1_outputs"
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


def locate_attachment1(root: Path) -> Path:
    candidates = [
        p
        for p in root.rglob("*.xlsx")
        if "preprocess_outputs" not in str(p)
        and "problem1_outputs" not in str(p)
        and ("附件1" in p.name or "问题1" in p.name)
    ]
    if not candidates:
        raise FileNotFoundError("未找到附件1 Excel 文件。")
    candidates = sorted(candidates, key=lambda p: (len(str(p)), p.name))
    return candidates[0]


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    return {
        "对象": name,
        "样本数": int(len(y_true)),
        "RMSE_mm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_mm": float(mean_absolute_error(y_true, y_pred)),
        "Bias_mm": float(np.mean(y_pred - y_true)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def blocked_cv(
    x: np.ndarray,
    y: np.ndarray,
    model,
    n_splits: int = 5,
) -> tuple[np.ndarray, pd.DataFrame]:
    n = len(y)
    fold_ids = np.array_split(np.arange(n), n_splits)
    pred = np.full(n, np.nan, dtype=float)
    rows: list[dict[str, float | int | str]] = []

    for i, test_idx in enumerate(fold_ids, start=1):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        fitted = clone(model)
        fitted.fit(x[train_idx].reshape(-1, 1), y[train_idx])
        fold_pred = fitted.predict(x[test_idx].reshape(-1, 1))
        pred[test_idx] = fold_pred
        row = metric_row(f"第{i}折", y[test_idx], fold_pred)
        row["起始样本序号"] = int(test_idx[0] + 1)
        row["终止样本序号"] = int(test_idx[-1] + 1)
        rows.append(row)

    rows.append(metric_row("5折汇总", y, pred))
    return pred, pd.DataFrame(rows)


def fit_model_comparison(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    candidates = {
        "原始A未校正": None,
        "普通线性回归": LinearRegression(),
        "Huber稳健线性回归": HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000),
        "二次多项式回归": make_pipeline(
            PolynomialFeatures(2, include_bias=False), LinearRegression()
        ),
    }
    rows: list[dict[str, float | str]] = []
    for name, model in candidates.items():
        if model is None:
            pred = x
        else:
            fitted = clone(model)
            fitted.fit(x.reshape(-1, 1), y)
            pred = fitted.predict(x.reshape(-1, 1))
        rows.append(metric_row(name, y, pred))
    return pd.DataFrame(rows)


def save_figures(
    paths: Paths,
    x_all: np.ndarray,
    y_all: np.ndarray,
    corrected: np.ndarray,
    time: pd.Series,
) -> None:
    order = np.argsort(x_all)
    plt.figure(figsize=(7.2, 5.2))
    plt.scatter(x_all, y_all, s=7, alpha=0.35, label="Observed A-B pairs")
    plt.plot(
        x_all[order],
        corrected[order],
        color="#c43c39",
        linewidth=2.0,
        label="Robust calibration curve",
    )
    plt.xlabel("A: fiber displacement (mm)")
    plt.ylabel("B: reference displacement (mm)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(paths.figures / "问题1_A_B散点与校正曲线.png", dpi=200)
    plt.close()

    raw_err = x_all - y_all
    cal_err = corrected - y_all
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(time, raw_err, linewidth=0.8, alpha=0.65, label="Before calibration")
    plt.plot(time, cal_err, linewidth=0.8, alpha=0.75, label="After calibration")
    plt.axhline(0, color="#222222", linewidth=0.8)
    plt.xlabel("Time")
    plt.ylabel("Residual to reference B (mm)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(paths.figures / "问题1_校正前后误差对比.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6.4, 4.6))
    plt.boxplot(
        [raw_err, cal_err],
        labels=["Before", "After"],
        showfliers=False,
        widths=0.45,
    )
    plt.ylabel("Residual to reference B (mm)")
    plt.tight_layout()
    plt.savefig(paths.figures / "问题1_校正前后残差箱线图.png", dpi=200)
    plt.close()


def main() -> None:
    root = Path.cwd()
    paths = make_paths(root)
    source = locate_attachment1(root)
    df = pd.read_excel(source)

    time_col = df.columns[0]
    numeric_cols = list(df.select_dtypes(include="number").columns)
    if len(numeric_cols) < 2:
        raise ValueError("附件1至少需要包含两列数值型位移数据。")
    a_col, b_col = numeric_cols[:2]

    time = pd.to_datetime(df[time_col])
    a = df[a_col].astype(float).to_numpy()
    b = df[b_col].astype(float).to_numpy()

    initial_zero = (np.arange(len(df)) == 0) & (a == 0) & (b == 0)
    any_zero = (a == 0) | (b == 0)
    non_initial_zero = any_zero & ~initial_zero
    modeling_mask = ~non_initial_zero

    x_fit = a[modeling_mask]
    y_fit = b[modeling_mask]

    final_model = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000)
    final_model.fit(x_fit.reshape(-1, 1), y_fit)
    corrected_all = final_model.predict(a.reshape(-1, 1))
    corrected_fit = corrected_all[modeling_mask]

    cv_pred, cv_table = blocked_cv(x_fit, y_fit, final_model, n_splits=5)
    cv_raw_table = pd.DataFrame([metric_row("原始A未校正", y_fit, x_fit)])
    fit_metrics = pd.DataFrame(
        [
            metric_row("校正前_全部样本", b, a),
            metric_row("校正后_全部样本", b, corrected_all),
            metric_row("校正前_有效样本", y_fit, x_fit),
            metric_row("校正后_有效样本", y_fit, corrected_fit),
        ]
    )
    comparison = fit_model_comparison(x_fit, y_fit)

    table_1_1 = pd.DataFrame(
        {
            "校正前数据x": TARGET_VALUES,
            "校正后数据y": final_model.predict(TARGET_VALUES.reshape(-1, 1)),
        }
    )
    table_1_1["校正后数据y_三位小数"] = table_1_1["校正后数据y"].round(3)

    result = df.copy()
    result["问题1_建模样本标记"] = modeling_mask
    result["问题1_非初始零值异常标记"] = non_initial_zero
    result["问题1_A校正后_mm"] = corrected_all
    result["问题1_校正前残差_A减B_mm"] = a - b
    result["问题1_校正后残差_校正A减B_mm"] = corrected_all - b
    result["问题1_校正前绝对误差_mm"] = np.abs(a - b)
    result["问题1_校正后绝对误差_mm"] = np.abs(corrected_all - b)

    zero_log = result.loc[
        non_initial_zero,
        [time_col, a_col, b_col, "问题1_非初始零值异常标记"],
    ].copy()
    zero_log.insert(0, "原始数据行号", zero_log.index + 2)

    save_figures(paths, a, b, corrected_all, time)

    excel_path = paths.data / "问题1_校正结果.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="校正明细", index=False)
        table_1_1.to_excel(writer, sheet_name="表1.1", index=False)
        fit_metrics.to_excel(writer, sheet_name="误差指标", index=False)
        cv_table.to_excel(writer, sheet_name="时间分块交叉验证", index=False)
        comparison.to_excel(writer, sheet_name="模型对照", index=False)
        zero_log.to_excel(writer, sheet_name="零值异常日志", index=False)

    result.to_csv(paths.data / "问题1_校正明细.csv", index=False, encoding="utf-8-sig")
    table_1_1.to_csv(paths.tables / "表1_1_校正结果.csv", index=False, encoding="utf-8-sig")
    fit_metrics.to_csv(paths.tables / "问题1_校正前后误差指标.csv", index=False, encoding="utf-8-sig")
    cv_table.to_csv(paths.tables / "问题1_时间分块交叉验证.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(paths.tables / "问题1_模型对照指标.csv", index=False, encoding="utf-8-sig")
    zero_log.to_csv(paths.logs / "问题1_非初始零值异常日志.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source_file": str(source),
        "rows": int(len(df)),
        "time_start": str(time.iloc[0]),
        "time_end": str(time.iloc[-1]),
        "time_interval_counts": {
            str(k): int(v) for k, v in time.diff().dropna().value_counts().items()
        },
        "a_column": a_col,
        "b_column": b_col,
        "missing_counts": {str(k): int(v) for k, v in df.isna().sum().items()},
        "initial_zero_kept": int(initial_zero.sum()),
        "non_initial_zero_excluded_from_fit": int(non_initial_zero.sum()),
        "modeling_rows": int(modeling_mask.sum()),
        "final_model": {
            "type": "HuberRegressor",
            "epsilon": 1.35,
            "alpha": 0.0,
            "intercept": float(final_model.intercept_),
            "slope": float(final_model.coef_[0]),
            "calibration_expression": f"y = {final_model.intercept_:.9f} + {final_model.coef_[0]:.9f} * x",
        },
        "outputs": {
            "excel": str(excel_path),
            "table_1_1": str(paths.tables / "表1_1_校正结果.csv"),
            "metrics": str(paths.tables / "问题1_校正前后误差指标.csv"),
            "cv": str(paths.tables / "问题1_时间分块交叉验证.csv"),
            "zero_log": str(paths.logs / "问题1_非初始零值异常日志.csv"),
            "figures": [str(p) for p in sorted(paths.figures.glob("问题1_*.png"))],
        },
    }

    with (paths.logs / "问题1_建模日志.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    text_log = [
        "问题1：光纤位移计数据A校正日志",
        f"原始文件：{source}",
        f"样本量：{len(df)}",
        f"时间范围：{time.iloc[0]} 至 {time.iloc[-1]}",
        f"保留初始零点数量：{int(initial_zero.sum())}",
        f"排除非初始零值记录数量：{int(non_initial_zero.sum())}",
        f"有效建模样本量：{int(modeling_mask.sum())}",
        "主模型：Huber稳健一元仿射校正",
        f"校正方程：y = {final_model.intercept_:.9f} + {final_model.coef_[0]:.9f} * x",
        "",
        "校正前后误差指标：",
        fit_metrics.to_string(index=False),
        "",
        "5折时间分块交叉验证：",
        cv_table.to_string(index=False),
        "",
        "表1.1校正结果：",
        table_1_1.to_string(index=False),
    ]
    (paths.logs / "问题1_建模日志.txt").write_text("\n".join(text_log), encoding="utf-8")

    print("Problem 1 calibration completed.")
    print(f"Source: {source}")
    print(f"Output directory: {paths.out}")
    print(f"Calibration: y = {final_model.intercept_:.9f} + {final_model.coef_[0]:.9f} * x")
    print(table_1_1.to_string(index=False))
    print(fit_metrics.to_string(index=False))
    print(cv_table.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
