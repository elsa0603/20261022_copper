from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "copper_foil_random_forest.joblib"
DATA_CANDIDATES = [
    ROOT / "copper_foil_transformed.csv",
    ROOT / "copper_foil_prepared.csv",
    ROOT / "copper_foil_1000.csv",
]

st.set_page_config(
    page_title="銅箔製程監控與異常預測",
    page_icon="🟠",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data(path_string: str) -> pd.DataFrame:
    """讀取並快取儀表板使用的 CSV 資料。"""
    data = pd.read_csv(path_string, encoding="utf-8-sig")
    if "Timestamp" in data.columns:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    return data


@st.cache_resource(show_spinner="正在載入 Random Forest 模型…")
def load_model(path_string: str):
    """載入訓練筆記本匯出的模型封裝。"""
    return joblib.load(path_string)


def get_data_path() -> Path | None:
    """依優先順序選擇可用的資料檔。"""
    return next((path for path in DATA_CANDIDATES if path.exists()), None)


def friendly_shift(column: str) -> str:
    """將 one-hot 欄位名稱轉為容易辨認的班別名稱。"""
    return column.removeprefix("Shift_")


def apply_dashboard_filters(df: pd.DataFrame) -> pd.DataFrame:
    """在左側側邊欄提供篩選條件，並回傳篩選後的資料。"""
    filtered = df.copy()

    date_bounds = None
    if "Timestamp" in df.columns:
        all_dates = df["Timestamp"].dropna()
        if not all_dates.empty:
            date_bounds = (all_dates.min().date(), all_dates.max().date())

    shift_columns = [col for col in df.columns if col.startswith("Shift_")]
    if "Shift" in df.columns:
        all_shift_options = sorted(df["Shift"].dropna().astype(str).unique().tolist())
    elif shift_columns:
        all_shift_values = df[shift_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        active_shifts = all_shift_values.idxmax(axis=1).str.removeprefix("Shift_")
        active_shifts = active_shifts[all_shift_values.sum(axis=1) > 0]
        all_shift_options = sorted(active_shifts.unique().tolist())
    else:
        all_shift_options = []

    thickness_bounds = None
    if "Thick_Mean_um" in df.columns:
        all_thickness = pd.to_numeric(df["Thick_Mean_um"], errors="coerce").dropna()
        if not all_thickness.empty:
            low, high = float(all_thickness.min()), float(all_thickness.max())
            thickness_bounds = (low - 0.01, high + 0.01) if low == high else (low, high)

    with st.sidebar.expander("條件篩選", expanded=True):
        if date_bounds:
            date_range = st.date_input(
                "日期範圍",
                value=date_bounds,
                min_value=date_bounds[0],
                max_value=date_bounds[1],
                key="dashboard_date_range",
            )
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                start_date, end_date = date_range
                filtered = filtered[
                    filtered["Timestamp"].dt.date.between(start_date, end_date)
                ]

        if all_shift_options:
            if "Shift" in filtered.columns:
                filtered["Shift_Filter"] = filtered["Shift"].fillna("未知").astype(str)
            else:
                shift_values = filtered[shift_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
                filtered["Shift_Filter"] = shift_values.idxmax(axis=1).str.removeprefix("Shift_")
                filtered.loc[shift_values.sum(axis=1) == 0, "Shift_Filter"] = "未知"
            selected_shifts = st.multiselect(
                "班別",
                options=all_shift_options,
                default=all_shift_options,
                key="dashboard_shifts",
            )
            filtered = filtered[filtered["Shift_Filter"].isin(selected_shifts)]

        if "Is_Anomaly" in filtered.columns:
            status = pd.to_numeric(filtered["Is_Anomaly"], errors="coerce").fillna(0).astype(int)
            status_choice = st.selectbox(
                "異常狀態",
                ["全部", "正常", "異常"],
                key="dashboard_anomaly_status",
            )
            if status_choice == "正常":
                filtered = filtered[status == 0]
            elif status_choice == "異常":
                filtered = filtered[status == 1]

        if thickness_bounds and not filtered.empty:
            thickness_range = st.slider(
                "平均厚度範圍 (μm)",
                min_value=thickness_bounds[0],
                max_value=thickness_bounds[1],
                value=thickness_bounds,
                key="dashboard_thickness_range",
            )
            values = pd.to_numeric(filtered["Thick_Mean_um"], errors="coerce")
            filtered = filtered[values.between(*thickness_range)]

    return filtered

def render_dashboard() -> None:
    st.title("銅箔製程資料視覺化")
    st.caption("檢視製程參數、厚度分布與異常標記。圖表使用專案中的處理後資料。")

    data_path = get_data_path()
    if data_path is None:
        st.error("找不到資料 CSV。請將 copper_foil_transformed.csv 或 copper_foil_1000.csv 放在 05_app.py 同一資料夾。")
        return

    try:
        df = load_data(str(data_path))
    except Exception as exc:
        st.error(f"讀取資料失敗：{exc}")
        return

    total_rows = len(df)
    df = apply_dashboard_filters(df)
    st.caption(f"資料來源：{data_path.name}｜顯示 {len(df):,} / {total_rows:,} 筆資料")
    if df.empty:
        st.warning("目前篩選條件沒有符合的資料，請調整左側條件。")
        return

    anomaly_count = int(pd.to_numeric(df["Is_Anomaly"], errors="coerce").fillna(0).sum()) if "Is_Anomaly" in df else None
    thickness_mean = pd.to_numeric(df["Thick_Mean_um"], errors="coerce").mean() if "Thick_Mean_um" in df else np.nan
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("資料筆數", f"{len(df):,}")
    kpi2.metric("異常筆數", f"{anomaly_count:,}" if anomaly_count is not None else "資料未提供")
    kpi3.metric("平均厚度", f"{thickness_mean:.2f} μm" if pd.notna(thickness_mean) else "資料未提供")

    st.subheader("平均厚度趨勢")
    if {"Timestamp", "Thick_Mean_um"}.issubset(df.columns):
        trend_df = df.dropna(subset=["Timestamp", "Thick_Mean_um"]).sort_values("Timestamp").copy()
        if "Is_Anomaly" in trend_df.columns:
            anomaly = pd.to_numeric(trend_df["Is_Anomaly"], errors="coerce").fillna(0).astype(int)
            trend_df["狀態"] = np.where(anomaly == 1, "異常", "正常")
            st.scatter_chart(
                trend_df,
                x="Timestamp",
                y="Thick_Mean_um",
                color="狀態",
                x_label="時間",
                y_label="平均厚度 (μm)",
                use_container_width=True,
            )
        else:
            st.line_chart(trend_df.set_index("Timestamp")[["Thick_Mean_um"]], y="Thick_Mean_um", y_label="平均厚度 (μm)")
        st.caption("目標厚度：18 μm")
    else:
        st.info("資料缺少 Timestamp 或 Thick_Mean_um 欄位，無法繪製厚度趨勢。")

    left, right = st.columns(2)
    with left:
        st.subheader("平均厚度分布")
        if "Thick_Mean_um" in df:
            hist_df = df[["Thick_Mean_um"]].copy()
            hist_df["厚度區間"] = pd.cut(pd.to_numeric(hist_df["Thick_Mean_um"], errors="coerce"), bins=20)
            if "Is_Anomaly" in df:
                status = np.where(pd.to_numeric(df["Is_Anomaly"], errors="coerce").fillna(0).astype(int) == 1, "異常", "正常")
                hist_df["狀態"] = status
                histogram = pd.crosstab(hist_df["厚度區間"], hist_df["狀態"])
            else:
                histogram = hist_df["厚度區間"].value_counts().sort_index().to_frame("筆數")
            histogram.index = histogram.index.astype(str)
            st.bar_chart(histogram, x_label="平均厚度區間", y_label="筆數")
        else:
            st.info("資料缺少 Thick_Mean_um 欄位。")

    with right:
        st.subheader("電流密度與平均厚度")
        if {"Current_ASD", "Thick_Mean_um"}.issubset(df.columns):
            scatter_df = df[["Current_ASD", "Thick_Mean_um"]].copy()
            if "Is_Anomaly" in df:
                scatter_df["異常標記"] = pd.to_numeric(df["Is_Anomaly"], errors="coerce").fillna(0).astype(str)
            st.scatter_chart(
                scatter_df,
                x="Current_ASD",
                y="Thick_Mean_um",
                color="異常標記" if "異常標記" in scatter_df else None,
                x_label="電流密度 (ASD)",
                y_label="平均厚度 (μm)",
                use_container_width=True,
            )
        else:
            st.info("資料缺少 Current_ASD 或 Thick_Mean_um 欄位。")

    st.subheader("主要特徵相關性")
    numeric_features = [
        "Line_Speed_mpm", "Current_ASD", "Bath_Temp_C", "Cu_Conc_gL",
        "Additive_ppm", "Jet_Pressure_bar", "Edge_Current_Bias",
        "Thick_Mean_um", "Thick_Range_um",
    ]
    available = [col for col in numeric_features if col in df.columns]
    if len(available) >= 2:
        corr = df[available].apply(pd.to_numeric, errors="coerce").corr()
        st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info("資料中的數值特徵不足，無法計算相關性。")

    with st.expander("查看資料列"):
        st.dataframe(df, use_container_width=True, hide_index=True)

def render_prediction() -> None:
    st.title("Random Forest 異常預測")
    st.caption("輸入可在製程中取得的參數，使用訓練好的模型估計異常風險。")

    if not MODEL_PATH.exists():
        st.error(f"找不到模型檔：{MODEL_PATH.name}。請先執行 03_random_forest_training.ipynb 產生模型。")
        return

    try:
        artifact = load_model(str(MODEL_PATH))
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
        threshold = float(artifact.get("threshold", 0.5))
        medians = artifact.get("imputation_medians", {})
    except Exception as exc:
        st.error(f"載入模型失敗：{exc}")
        return

    st.info(f"模型目標：{artifact.get('target_column', 'Is_Anomaly')}｜預測閾值：{threshold:.2f}")
    input_values: dict[str, float] = {}
    shift_columns = [col for col in feature_columns if col.startswith("Shift_")]
    shift_labels = [friendly_shift(col) for col in shift_columns]

    with st.form("prediction_form"):
        st.subheader("製程參數")
        col1, col2 = st.columns(2)
        numeric_fields = [
            ("Line_Speed_mpm", "線速 (m/min)", 3.20, 0.01, 0.0, 10.0),
            ("Current_ASD", "電流密度 (ASD)", 58.0, 0.1, 0.0, 100.0),
            ("Bath_Temp_C", "槽液溫度 (°C)", 50.0, 0.1, 0.0, 100.0),
            ("Cu_Conc_gL", "銅離子濃度 (g/L)", 88.0, 0.1, 0.0, 200.0),
            ("Additive_ppm", "添加劑濃度 (ppm)", 3.5, 0.01, 0.0, 100.0),
            ("Jet_Pressure_bar", "噴流壓力 (bar)", 2.5, 0.01, 0.0, 20.0),
            ("Edge_Current_Bias", "邊緣電流偏差", 0.0, 0.01, -10.0, 10.0),
            ("Hour", "小時 (0–23)", 12, 1, 0, 23),
            ("DayOfWeek", "星期 (週一=0) ", 0, 1, 0, 6),
        ]
        for index, (key, label, default, step, min_value, max_value) in enumerate(numeric_fields):
            if key in feature_columns:
                with (col1 if index % 2 == 0 else col2):
                    input_values[key] = st.number_input(
                        label, min_value=min_value, max_value=max_value,
                        value=default, step=step, key=f"input_{key}",
                    )

        if shift_columns:
            selected_label = st.selectbox("班別", shift_labels, key="input_shift")
            selected_shift = shift_columns[shift_labels.index(selected_label)]
            for col in shift_columns:
                input_values[col] = int(col == selected_shift)

        if "Is_Weekend" in feature_columns and "DayOfWeek" in input_values:
            input_values["Is_Weekend"] = int(input_values["DayOfWeek"] >= 5)

        submitted = st.form_submit_button("執行異常預測", type="primary", use_container_width=True)

    if submitted:
        # 未在表單中出現的特徵以訓練階段保存的中位數補上，並依模型要求排序。
        input_row = {col: input_values.get(col, medians.get(col, 0)) for col in feature_columns}
        X_input = pd.DataFrame([input_row], columns=feature_columns)
        try:
            predicted_class = int(model.predict(X_input)[0])
            class_list = list(model.classes_)
            probabilities = model.predict_proba(X_input)[0]
            anomaly_probability = float(probabilities[class_list.index(1)])
        except Exception as exc:
            st.error(f"預測失敗：{exc}")
            return

        st.divider()
        st.subheader("預測結果")
        result_col, probability_col = st.columns(2)
        with result_col:
            if predicted_class == 1:
                st.error("模型預測：異常")
            else:
                st.success("模型預測：正常")
        with probability_col:
            st.metric("異常機率", f"{anomaly_probability:.1%}")
        st.progress(min(max(anomaly_probability, 0.0), 1.0), text=f"異常機率 {anomaly_probability:.1%}（分類閾值 {threshold:.0%}）")
        with st.expander("查看送入模型的特徵"):
            st.dataframe(X_input, use_container_width=True, hide_index=True)


with st.sidebar:
    st.title("銅箔製程分析")
    page = st.radio("功能分頁", ["資料視覺化", "異常預測"], index=0)
    st.divider()
    st.caption("資料視覺化可檢視製程與厚度；異常預測使用已訓練的 Random Forest 模型。")

if page == "資料視覺化":
    render_dashboard()
else:
    render_prediction()