from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ATTACH_DIR = BASE_DIR / "C 附件(Attachment)"
OUT_DIR = BASE_DIR / "preprocess_outputs"
DATA_DIR = OUT_DIR / "data"
LOG_DIR = OUT_DIR / "logs"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_DIR = OUT_DIR / "summary"

FILES = {
    "attachment1": "附件1：两组位移时序数据-问题1.xlsx",
    "attachment2": "附件2：位移时序数据-问题2.xlsx",
    "attachment3": "附件3：监测数据（训练集与实验集）-问题3.xlsx",
    "attachment4": "附件4：监测数据（训练集与实验集）-问题4.xlsx",
    "attachment5": "附件5：监测数据-问题5.xlsx",
}

LABEL_MAP = {
    "数据A_光纤位移计数据_mm": "A_fiber_mm",
    "数据B_振弦式位移计数据_mm": "B_vibrating_wire_mm",
    "A减B差值_mm": "A_minus_B_mm",
    "表面位移_mm": "surface_disp_mm",
    "降雨量_mm": "rainfall_mm",
    "孔隙水压力_kPa": "pore_pressure_kPa",
    "微震事件数": "microseismic_count",
    "干湿入渗系数": "dry_wet_infiltration",
    "阶段标签": "stage_label",
    "a_降雨量_mm": "a_rainfall_mm",
    "b_孔隙水压力_kPa": "b_pore_pressure_kPa",
    "c_微震事件数": "c_microseismic_count",
    "d_深部位移_mm": "d_deep_disp_mm",
    "e_表面位移_mm": "e_surface_disp_mm",
    "d_深部位移_mm": "d_deep_disp_mm",
    "e_表面位移_mm": "e_surface_disp_mm",
}


def label(col: str) -> str:
    return LABEL_MAP.get(col, col.encode("ascii", errors="ignore").decode("ascii") or "value")


def ensure_dirs() -> None:
    for path in [DATA_DIR, LOG_DIR, FIG_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return (
        name.replace("：", "_")
        .replace("（", "_")
        .replace("）", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def to_datetime_col(df: pd.DataFrame, col: str = "时间") -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def time_quality(df: pd.DataFrame, dataset: str, col: str = "时间") -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    t = to_datetime_col(df, col)
    diff_min = t.diff().dt.total_seconds().div(60)
    rows = []
    rows.extend(
        {
            "dataset": dataset,
            "row_index": int(i),
            "time": df.loc[i, col],
            "issue": "时间解析失败",
        }
        for i in df.index[t.isna()]
    )
    bad_gap = diff_min.notna() & (diff_min != 10)
    rows.extend(
        {
            "dataset": dataset,
            "row_index": int(i),
            "time": df.loc[i, col],
            "issue": f"时间间隔非10分钟: {diff_min.loc[i]}",
        }
        for i in df.index[bad_gap]
    )
    duplicated = t.duplicated(keep=False) & t.notna()
    rows.extend(
        {
            "dataset": dataset,
            "row_index": int(i),
            "time": df.loc[i, col],
            "issue": "时间重复",
        }
        for i in df.index[duplicated]
    )
    return pd.DataFrame(rows)


def id_quality(df: pd.DataFrame, dataset: str, col: str = "编号") -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    ids = df[col]
    rows = []
    duplicated = ids.duplicated(keep=False)
    rows.extend(
        {
            "dataset": dataset,
            "row_index": int(i),
            "id": ids.loc[i],
            "issue": "编号重复",
        }
        for i in df.index[duplicated]
    )
    if ids.notna().all() and len(ids) > 0:
        expected = np.arange(int(ids.iloc[0]), int(ids.iloc[0]) + len(ids))
        mismatch = ids.to_numpy() != expected
        rows.extend(
            {
                "dataset": dataset,
                "row_index": int(i),
                "id": ids.loc[i],
                "issue": "编号不连续",
            }
            for i in df.index[mismatch]
        )
    return pd.DataFrame(rows)


def add_missing_flags(df: pd.DataFrame, cols: Iterable[str], prefix: str = "缺失标记") -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[f"{prefix}_{col}"] = df[col].isna().astype(int)
    return df


def missing_log(df: pd.DataFrame, dataset: str, cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        mask = df[col].isna()
        for i in df.index[mask]:
            rows.append({"dataset": dataset, "row_index": int(i), "column": col, "issue": "缺失"})
    return pd.DataFrame(rows)


def physical_log(df: pd.DataFrame, dataset: str, nonnegative_cols: Iterable[str], integer_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in nonnegative_cols:
        if col not in df.columns:
            continue
        mask = df[col].notna() & (df[col] < 0)
        for i in df.index[mask]:
            rows.append(
                {
                    "dataset": dataset,
                    "row_index": int(i),
                    "column": col,
                    "value": df.loc[i, col],
                    "issue": "负值",
                }
            )
    for col in integer_cols:
        if col not in df.columns:
            continue
        ser = df[col]
        mask = ser.notna() & (np.abs(ser - np.round(ser)) > 1e-9)
        for i in df.index[mask]:
            rows.append(
                {
                    "dataset": dataset,
                    "row_index": int(i),
                    "column": col,
                    "value": df.loc[i, col],
                    "issue": "计数变量非整数",
                }
            )
    return pd.DataFrame(rows)


def robust_spike_flags(series: pd.Series, window: int = 9, z_thresh: float = 6.0, quantile: float = 0.995) -> pd.Series:
    ser = pd.to_numeric(series, errors="coerce")
    diff = ser.diff()
    med = diff.rolling(window, center=True, min_periods=max(3, window // 2)).median()
    mad = (diff - med).abs().rolling(window, center=True, min_periods=max(3, window // 2)).median()
    robust_scale = 1.4826 * mad.replace(0, np.nan)
    score = (diff - med).abs() / robust_scale
    finite_score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if finite_score.empty:
        return pd.Series(0, index=series.index, dtype=int)
    cutoff = max(z_thresh, float(finite_score.quantile(quantile)))
    return (score > cutoff).fillna(False).astype(int)


def add_zero_suspicion(df: pd.DataFrame, cols: Iterable[str], dataset: str) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        flag_col = f"可疑零值标记_{col}"
        zero = (df[col] == 0).fillna(False)
        if len(df) > 0:
            zero.iloc[0] = False
        df[flag_col] = zero.astype(int)
        for i in df.index[zero]:
            rows.append(
                {
                    "dataset": dataset,
                    "row_index": int(i),
                    "column": col,
                    "value": df.loc[i, col],
                    "issue": "非起始位置零值",
                }
            )
    return pd.DataFrame(rows)


def add_spike_suspicion(df: pd.DataFrame, cols: Iterable[str], dataset: str) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        flag = robust_spike_flags(df[col])
        df[f"突变候选标记_{col}"] = flag
        for i in df.index[flag.astype(bool)]:
            rows.append(
                {
                    "dataset": dataset,
                    "row_index": int(i),
                    "column": col,
                    "value": df.loc[i, col],
                    "issue": "局部差分突变候选",
                }
            )
    return pd.DataFrame(rows)


def add_blast_flags(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    dist = "爆破点距离_m"
    charge = "单段最大药量_kg"
    if dist not in df.columns or charge not in df.columns:
        return pd.DataFrame()
    dist_nonnull = df[dist].notna()
    charge_nonnull = df[charge].notna()
    df["是否爆破"] = (dist_nonnull | charge_nonnull).astype(int)
    df["爆破字段冲突标记"] = (dist_nonnull ^ charge_nonnull).astype(int)
    rows = []
    for i in df.index[df["爆破字段冲突标记"].astype(bool)]:
        rows.append(
            {
                "dataset": dataset,
                "row_index": int(i),
                "time": df.loc[i, "时间"] if "时间" in df.columns else None,
                "爆破点距离_m": df.loc[i, dist],
                "单段最大药量_kg": df.loc[i, charge],
                "issue": "爆破距离与药量一空一非空",
            }
        )
    return pd.DataFrame(rows)


def add_time_from_id(df: pd.DataFrame, start: str) -> None:
    if "编号" not in df.columns:
        return
    start_time = pd.Timestamp(start)
    df["推算时间"] = start_time + pd.to_timedelta((df["编号"] - 1) * 10, unit="min")


def add_velocity_cols(df: pd.DataFrame, value_col: str, time_col: str | None = None) -> None:
    if value_col not in df.columns:
        return
    interval_h = 10 / 60
    if time_col and time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
        dt = t.diff().dt.total_seconds().div(3600)
        dt = dt.where(dt > 0, interval_h)
    else:
        dt = pd.Series(interval_h, index=df.index)
    df[f"原始增量_{value_col}"] = df[value_col].diff()
    df[f"原始速度_{value_col}_mm_h"] = df[f"原始增量_{value_col}"] / dt
    df[f"原始加速度_{value_col}_mm_h2"] = df[f"原始速度_{value_col}_mm_h"].diff() / dt


def save_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)


def save_log(name: str, frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame()
    out.to_csv(LOG_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    return out


def line_plot(df: pd.DataFrame, cols: list[str], path: Path, title: str, x_col: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = pd.to_datetime(df[x_col], errors="coerce") if x_col and x_col in df.columns else np.arange(len(df))
    for col in cols:
        if col in df.columns:
            ax.plot(x, df[col], linewidth=0.8, label=label(col))
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df[x_col], df[y_col], s=8, alpha=0.5)
    ax.set_xlabel(label(x_col))
    ax.set_ylabel(label(y_col))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def missing_bar(df: pd.DataFrame, cols: list[str], path: Path, title: str) -> None:
    counts = df[cols].isna().sum()
    counts.index = [label(c) for c in counts.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("missing count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def blast_event_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    if "是否爆破" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 2.8))
    x = pd.to_datetime(df["时间"], errors="coerce") if "时间" in df.columns else np.arange(len(df))
    ax.vlines(x[df["是否爆破"].astype(bool)], 0, 1, linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_yticks([0, 1])
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_df(df: pd.DataFrame, dataset: str) -> list[dict[str, object]]:
    rows = []
    for col in df.columns:
        ser = df[col]
        row = {
            "dataset": dataset,
            "column": col,
            "rows": len(df),
            "dtype": str(ser.dtype),
            "missing": int(ser.isna().sum()),
            "non_null": int(ser.notna().sum()),
        }
        if pd.api.types.is_numeric_dtype(ser):
            non_na = ser.dropna()
            row.update(
                {
                    "min": float(non_na.min()) if len(non_na) else math.nan,
                    "max": float(non_na.max()) if len(non_na) else math.nan,
                    "zero_count": int((ser == 0).sum()),
                    "negative_count": int((ser < 0).sum()),
                }
            )
        rows.append(row)
    return rows


def process_attachment1(summary_rows: list[dict[str, object]]) -> None:
    path = ATTACH_DIR / FILES["attachment1"]
    df = pd.read_excel(path, sheet_name="Sheet1")
    dataset = "附件1_Sheet1"
    df["时间解析"] = to_datetime_col(df)
    df["A减B差值_mm"] = df["数据A_光纤位移计数据_mm"] - df["数据B_振弦式位移计数据_mm"]
    df["A相对B偏差"] = df["A减B差值_mm"] / df["数据B_振弦式位移计数据_mm"].replace(0, np.nan)
    logs = [
        time_quality(df, dataset, "时间"),
        add_zero_suspicion(df, ["数据A_光纤位移计数据_mm", "数据B_振弦式位移计数据_mm"], dataset),
        add_spike_suspicion(df, ["数据A_光纤位移计数据_mm", "数据B_振弦式位移计数据_mm"], dataset),
        physical_log(
            df,
            dataset,
            ["数据A_光纤位移计数据_mm", "数据B_振弦式位移计数据_mm"],
            [],
        ),
    ]
    save_xlsx(DATA_DIR / "附件1_保守预处理.xlsx", {"Sheet1": df})
    save_log("附件1_质量日志", logs)
    line_plot(
        df,
        ["数据A_光纤位移计数据_mm", "数据B_振弦式位移计数据_mm"],
        FIG_DIR / "附件1_A_B时序.png",
        "Attachment 1 displacement series",
        "时间",
    )
    line_plot(df, ["A减B差值_mm"], FIG_DIR / "附件1_A减B差值.png", "Attachment 1 A minus B", "时间")
    scatter_plot(
        df,
        "数据A_光纤位移计数据_mm",
        "数据B_振弦式位移计数据_mm",
        FIG_DIR / "附件1_A_B散点.png",
        "Attachment 1 A-B scatter",
    )
    summary_rows.extend(summarize_df(df, dataset))


def process_attachment2(summary_rows: list[dict[str, object]]) -> None:
    path = ATTACH_DIR / FILES["attachment2"]
    df = pd.read_excel(path, sheet_name="Sheet1")
    dataset = "附件2_Sheet1"
    add_time_from_id(df, "2024-05-04 00:00")
    add_velocity_cols(df, "表面位移_mm", "推算时间")
    logs = [
        id_quality(df, dataset),
        add_zero_suspicion(df, ["表面位移_mm"], dataset),
        add_spike_suspicion(df, ["表面位移_mm", "原始速度_表面位移_mm_mm_h"], dataset),
        physical_log(df, dataset, ["表面位移_mm"], []),
    ]
    save_xlsx(DATA_DIR / "附件2_保守预处理.xlsx", {"Sheet1": df})
    save_log("附件2_质量日志", logs)
    line_plot(df, ["表面位移_mm"], FIG_DIR / "附件2_位移时序.png", "Attachment 2 displacement", "推算时间")
    line_plot(
        df,
        ["原始速度_表面位移_mm_mm_h"],
        FIG_DIR / "附件2_原始速度.png",
        "Attachment 2 raw velocity",
        "推算时间",
    )
    line_plot(
        df,
        ["原始加速度_表面位移_mm_mm_h2"],
        FIG_DIR / "附件2_原始加速度.png",
        "Attachment 2 raw acceleration",
        "推算时间",
    )
    summary_rows.extend(summarize_df(df, dataset))


def process_attachment3(summary_rows: list[dict[str, object]]) -> None:
    path = ATTACH_DIR / FILES["attachment3"]
    train = pd.read_excel(path, sheet_name="训练集")
    test = pd.read_excel(path, sheet_name="实验集")
    train_rename = {
        "a:降雨量_mm": "a_降雨量_mm",
        "b:孔隙水压力_kPa": "b_孔隙水压力_kPa",
        "c:微震事件数": "c_微震事件数",
        "d:深部位移_mm": "d_深部位移_mm",
        "e:表面位移_mm": "e_表面位移_mm",
    }
    test_rename = {
        "降雨量_mm": "a_降雨量_mm",
        "孔隙水压力_kPa": "b_孔隙水压力_kPa",
        "微震事件数": "c_微震事件数",
        "深部位移_mm": "d_深部位移_mm",
        "表面位移_mm": "e_表面位移_mm",
    }
    train = train.rename(columns=train_rename)
    test = test.rename(columns=test_rename)
    cols = ["a_降雨量_mm", "b_孔隙水压力_kPa", "c_微震事件数", "d_深部位移_mm", "e_表面位移_mm"]
    add_missing_flags(train, cols)
    add_missing_flags(test, cols)
    train_logs = [
        id_quality(train, "附件3_训练集"),
        missing_log(train, "附件3_训练集", cols),
        physical_log(train, "附件3_训练集", cols, ["c_微震事件数"]),
        add_spike_suspicion(train, ["b_孔隙水压力_kPa", "d_深部位移_mm", "e_表面位移_mm"], "附件3_训练集"),
    ]
    test_explanatory = ["a_降雨量_mm", "b_孔隙水压力_kPa", "c_微震事件数", "d_深部位移_mm"]
    test_logs = [
        id_quality(test, "附件3_实验集"),
        missing_log(test, "附件3_实验集", test_explanatory),
        physical_log(test, "附件3_实验集", cols, ["c_微震事件数"]),
        add_spike_suspicion(test, ["b_孔隙水压力_kPa", "d_深部位移_mm"], "附件3_实验集"),
    ]
    save_xlsx(DATA_DIR / "附件3_保守预处理.xlsx", {"训练集": train, "实验集": test})
    save_log("附件3_质量日志", train_logs + test_logs)
    missing_bar(train, cols, FIG_DIR / "附件3_训练集缺失统计.png", "Attachment 3 train missing")
    missing_bar(test, cols, FIG_DIR / "附件3_实验集缺失统计.png", "Attachment 3 test missing")
    line_plot(train, cols, FIG_DIR / "附件3_训练集变量时序.png", "Attachment 3 train variables")
    scatter_plot(train.dropna(subset=["d_深部位移_mm", "e_表面位移_mm"]), "d_深部位移_mm", "e_表面位移_mm", FIG_DIR / "附件3_深部表面散点.png", "Deep-surface scatter")
    summary_rows.extend(summarize_df(train, "附件3_训练集"))
    summary_rows.extend(summarize_df(test, "附件3_实验集"))


def process_attachment4(summary_rows: list[dict[str, object]]) -> None:
    path = ATTACH_DIR / FILES["attachment4"]
    train = pd.read_excel(path, sheet_name="训练集")
    test = pd.read_excel(path, sheet_name="实验集")
    for df, dataset in [(train, "附件4_训练集"), (test, "附件4_实验集")]:
        df["时间解析"] = to_datetime_col(df)
        add_blast_flags(df, dataset)
        add_velocity_cols(df, "表面位移_mm", "时间")
    logs = [
        time_quality(train, "附件4_训练集"),
        time_quality(test, "附件4_实验集"),
        add_blast_flags(train, "附件4_训练集"),
        add_blast_flags(test, "附件4_实验集"),
        physical_log(
            train,
            "附件4_训练集",
            ["表面位移_mm", "降雨量_mm", "孔隙水压力_kPa", "微震事件数", "爆破点距离_m", "单段最大药量_kg"],
            ["微震事件数"],
        ),
        physical_log(
            test,
            "附件4_实验集",
            ["降雨量_mm", "孔隙水压力_kPa", "微震事件数", "爆破点距离_m", "单段最大药量_kg"],
            ["微震事件数", "阶段标签"],
        ),
        add_spike_suspicion(train, ["表面位移_mm", "孔隙水压力_kPa"], "附件4_训练集"),
    ]
    save_xlsx(DATA_DIR / "附件4_保守预处理.xlsx", {"训练集": train, "实验集": test})
    save_log("附件4_质量日志", logs)
    line_plot(train, ["表面位移_mm"], FIG_DIR / "附件4_训练集表面位移.png", "Attachment 4 train displacement", "时间")
    line_plot(test, ["阶段标签"], FIG_DIR / "附件4_实验集阶段标签.png", "Attachment 4 test stage label", "时间")
    blast_event_plot(train, FIG_DIR / "附件4_训练集爆破事件.png", "Attachment 4 train blast events")
    blast_event_plot(test, FIG_DIR / "附件4_实验集爆破事件.png", "Attachment 4 test blast events")
    summary_rows.extend(summarize_df(train, "附件4_训练集"))
    summary_rows.extend(summarize_df(test, "附件4_实验集"))


def process_attachment5(summary_rows: list[dict[str, object]]) -> None:
    path = ATTACH_DIR / FILES["attachment5"]
    df = pd.read_excel(path, sheet_name="Sheet1")
    dataset = "附件5_Sheet1"
    df["时间解析"] = to_datetime_col(df)
    add_blast_flags(df, dataset)
    add_velocity_cols(df, "表面位移_mm", "时间")
    logs = [
        time_quality(df, dataset),
        add_blast_flags(df, dataset),
        physical_log(
            df,
            dataset,
            ["表面位移_mm", "降雨量_mm", "孔隙水压力_kPa", "微震事件数", "干湿入渗系数", "爆破点距离_m", "单段最大药量_kg"],
            ["微震事件数"],
        ),
        add_zero_suspicion(df, ["表面位移_mm"], dataset),
        add_spike_suspicion(df, ["表面位移_mm", "原始速度_表面位移_mm_mm_h", "孔隙水压力_kPa"], dataset),
    ]
    save_xlsx(DATA_DIR / "附件5_保守预处理.xlsx", {"Sheet1": df})
    save_log("附件5_质量日志", logs)
    line_plot(df, ["表面位移_mm"], FIG_DIR / "附件5_表面位移.png", "Attachment 5 displacement", "时间")
    line_plot(df, ["原始速度_表面位移_mm_mm_h"], FIG_DIR / "附件5_原始速度.png", "Attachment 5 raw velocity", "时间")
    blast_event_plot(df, FIG_DIR / "附件5_爆破事件.png", "Attachment 5 blast events")
    line_plot(
        df,
        ["降雨量_mm", "孔隙水压力_kPa", "微震事件数", "干湿入渗系数"],
        FIG_DIR / "附件5_候选变量时序.png",
        "Attachment 5 candidate variables",
        "时间",
    )
    summary_rows.extend(summarize_df(df, dataset))


def main() -> None:
    ensure_dirs()
    summary_rows: list[dict[str, object]] = []
    process_attachment1(summary_rows)
    process_attachment2(summary_rows)
    process_attachment3(summary_rows)
    process_attachment4(summary_rows)
    process_attachment5(summary_rows)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_DIR / "预处理字段汇总.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(SUMMARY_DIR / "预处理字段汇总.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="字段汇总", index=False)


if __name__ == "__main__":
    main()
