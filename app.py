# 7
# streamlit run app.py

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Travel Recommendation System",
    layout="wide"
)

st.title("旅遊地點推薦系統")
st.caption("Unsupervised Clustering + Cluster-based Recommendation Demo")


# =========================
# Load Data
# =========================

@st.cache_data
def load_csv_data():

    user_df = pd.read_csv("data/travel_behavior_soft_clustered.csv")
    attractions = pd.read_csv("data/attractions.csv")

    method_comparison = None

    if os.path.exists("data/recommendation_method_comparison.csv"):

        method_comparison = pd.read_csv(
            "data/recommendation_method_comparison.csv"
        )

    cluster_interpretation = None

    if os.path.exists("data/soft_cluster_interpretation.csv"):

        cluster_interpretation = pd.read_csv(
            "data/soft_cluster_interpretation.csv"
        )

    return (
        user_df,
        attractions,
        method_comparison,
        cluster_interpretation
    )


@st.cache_resource
def load_cluster_model():

    return joblib.load("models/soft_cluster_pipeline.pkl")


(
    user_df,
    attractions,
    method_comparison,
    cluster_interpretation
) = load_csv_data()

cluster_model = load_cluster_model()


# =========================
# Basic Settings
# =========================

behavior_features = [
    "selected_indoor_score",
    "selected_cost_level",
    "selected_photo_value",
    "selected_nature_value",
    "selected_culture_value",
    "selected_food_value",
    "selected_popularity"
]

attraction_features = [
    "indoor_score",
    "cost_level",
    "photo_value",
    "nature_value",
    "culture_value",
    "food_value",
    "popularity"
]


# =========================
# Helper Functions
# =========================

def cosine_similarity(a, b):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def get_cluster_only_recommendations(
    cluster_id,
    top_k,
    cluster_behavior,
    attractions_df
):

    cluster_vec = cluster_behavior.loc[cluster_id].values

    rows = []

    for _, spot in attractions_df.iterrows():

        spot_vec = spot[attraction_features].values

        score = cosine_similarity(
            cluster_vec,
            spot_vec
        )

        rows.append({
            "name": spot["name"],
            "attraction_type": spot["attraction_type"],
            "cluster_score": score,
            "recommendation_score": score,
            "cost_level": spot["cost_level"],
            "indoor_score": spot["indoor_score"],
            "photo_value": spot["photo_value"],
            "nature_value": spot["nature_value"],
            "culture_value": spot["culture_value"],
            "food_value": spot["food_value"],
            "popularity": spot["popularity"],
            "estimated_time": spot["estimated_time"]
        })

    result = pd.DataFrame(rows)

    return (
        result
        .sort_values(
            "recommendation_score",
            ascending=False
        )
        .head(top_k)
    )


def predict_soft_cluster(input_data, cluster_model):

    features = cluster_model["features"]
    imputer = cluster_model["imputer"]
    scaler = cluster_model["scaler"]
    gmm = cluster_model["gmm"]

    input_df = pd.DataFrame([input_data])

    missing_features = [
        feature for feature in features
        if feature not in input_df.columns
    ]

    if len(missing_features) > 0:

        raise ValueError(
            "模型需要的欄位不存在，請確認 soft_cluster_pipeline.pkl 是否為新版。"
            f" 缺少欄位：{missing_features}"
        )

    X_input = input_df[features]

    X_imputed = imputer.transform(X_input)
    X_scaled = scaler.transform(X_imputed)

    cluster_probs = gmm.predict_proba(X_scaled)[0]

    predicted_cluster = int(np.argmax(cluster_probs))
    cluster_confidence = float(np.max(cluster_probs))

    return predicted_cluster, cluster_confidence, cluster_probs


# =========================
# Cluster Behavior Prototype
# =========================

cluster_behavior = (
    user_df
    .groupby("soft_cluster")[behavior_features]
    .mean()
)


# =========================
# Sidebar
# =========================

st.sidebar.header("系統設定")

top_k = st.sidebar.slider(
    "顯示推薦數量",
    min_value=1,
    max_value=10,
    value=5
)

st.sidebar.header("使用者輸入方式")

manual_mode = st.sidebar.checkbox(
    "使用手動輸入模式",
    value=True
)


# =========================
# Manual Input Mode
# =========================

input_data = None
selected_user = None
predicted_cluster = None
cluster_confidence = None
cluster_probs = None

if manual_mode:

    st.sidebar.header("手動輸入使用者條件")

    input_data = {}

    input_data["budget"] = st.sidebar.number_input(
        "Budget",
        min_value=100.0,
        max_value=5000.0,
        value=800.0,
        step=100.0
    )

    input_data["available_time"] = st.sidebar.number_input(
        "Available Time",
        min_value=1.0,
        max_value=12.0,
        value=5.0,
        step=0.5
    )

    input_data["weather_badness"] = st.sidebar.slider(
        "Weather Badness",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05
    )

    input_data["social_context"] = st.sidebar.slider(
        "Social Context",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    input_data["fatigue"] = st.sidebar.slider(
        "Fatigue",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1
    )

    input_data["spontaneity"] = st.sidebar.slider(
        "Spontaneity",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1
    )

    input_data["selected_indoor_score"] = st.sidebar.slider(
        "Indoor Preference",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1
    )

    input_data["selected_cost_level"] = st.sidebar.slider(
        "Cost Level",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1
    )

    input_data["selected_photo_value"] = st.sidebar.slider(
        "Photo Value",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1
    )

    input_data["selected_nature_value"] = st.sidebar.slider(
        "Nature Value",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1
    )

    input_data["selected_culture_value"] = st.sidebar.slider(
        "Culture Value",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1
    )

    input_data["selected_food_value"] = st.sidebar.slider(
        "Food Value",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1
    )

    input_data["selected_popularity"] = st.sidebar.slider(
        "Popularity",
        min_value=0.0,
        max_value=5.0,
        value=2.5,
        step=0.1
    )

    try:

        (
            predicted_cluster,
            cluster_confidence,
            cluster_probs
        ) = predict_soft_cluster(
            input_data,
            cluster_model
        )

        selected_user = pd.Series(input_data)
        selected_user["soft_cluster"] = predicted_cluster
        selected_user["soft_cluster_confidence"] = cluster_confidence

    except Exception as e:

        st.error(str(e))
        st.stop()

else:

    st.sidebar.header("既有使用者選擇")

    user_ids = sorted(user_df["user_id"].unique())

    selected_user_id = st.sidebar.selectbox(
        "選擇使用者 ID",
        user_ids
    )

    selected_user = user_df[
        user_df["user_id"] == selected_user_id
    ].iloc[0]

    predicted_cluster = int(selected_user["soft_cluster"])
    cluster_confidence = float(
        selected_user["soft_cluster_confidence"]
    )


# =========================
# User Info
# =========================

st.subheader("使用者資訊")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Soft Cluster",
        int(selected_user["soft_cluster"])
    )

with col2:
    st.metric(
        "Cluster Confidence",
        f"{selected_user['soft_cluster_confidence']:.3f}"
    )

with col3:
    st.metric(
        "Budget",
        f"{selected_user['budget']:.1f}"
    )

with col4:
    st.metric(
        "Available Time",
        f"{selected_user['available_time']:.1f}"
    )


if manual_mode and cluster_probs is not None:

    st.markdown("### Soft Cluster Probability")

    prob_df = pd.DataFrame({
        "cluster": list(range(len(cluster_probs))),
        "probability": cluster_probs
    })

    st.dataframe(
        prob_df,
        width="stretch"
    )


st.markdown("### 使用者輸入 / 行為特徵")

display_user_cols = [
    "weather_badness",
    "social_context",
    "fatigue",
    "spontaneity",
    "selected_indoor_score",
    "selected_cost_level",
    "selected_photo_value",
    "selected_nature_value",
    "selected_culture_value",
    "selected_food_value",
    "selected_popularity"
]

available_display_user_cols = [
    col for col in display_user_cols
    if col in selected_user.index
]

st.dataframe(
    selected_user[available_display_user_cols]
    .to_frame(name="value"),
    width="stretch",
    height=450
)


# =========================
# Cluster-based Recommendation
# =========================

st.subheader("Cluster-based 推薦景點")

st.info(
    "目前推薦只使用使用者所屬 cluster 的平均行為特徵，"
    "不直接用個人行為向量排序。"
)

cluster_id = int(selected_user["soft_cluster"])

if cluster_id not in cluster_behavior.index:

    st.warning("目前 cluster 不存在於 cluster_behavior 中，請重新確認模型與資料是否一致。")
    st.stop()

user_recs = get_cluster_only_recommendations(
    cluster_id=cluster_id,
    top_k=top_k,
    cluster_behavior=cluster_behavior,
    attractions_df=attractions
)

if len(user_recs) == 0:

    st.warning("此 cluster 沒有推薦結果。")

else:

    display_cols = [
        "name",
        "attraction_type",
        "recommendation_score",
        "cluster_score",
        "cost_level",
        "indoor_score",
        "photo_value",
        "nature_value",
        "culture_value",
        "food_value",
        "popularity",
        "estimated_time"
    ]

    display_cols = [
        col for col in display_cols
        if col in user_recs.columns
    ]

    st.dataframe(
        user_recs[display_cols],
        width="stretch"
    )

    st.markdown("### 推薦結果解釋")

    for _, row in user_recs.iterrows():

        with st.expander(f"{row['name']}"):

            st.write(f"景點類型：{row.get('attraction_type', 'N/A')}")
            st.write(f"Cluster Score：{row['cluster_score']:.4f}")
            st.write(
                "解釋：此推薦是將景點特徵向量與該 cluster 的平均行為特徵向量進行相似度比較。"
            )


# =========================
# Cluster Interpretation
# =========================

st.subheader("Cluster 解釋")

cluster_id = int(selected_user["soft_cluster"])

if cluster_interpretation is not None:

    cluster_info = cluster_interpretation[
        cluster_interpretation["cluster"] == cluster_id
    ]

    if len(cluster_info) > 0:

        st.write(f"目前使用者屬於 Cluster {cluster_id}")

        high_features = cluster_info[
            cluster_info["direction"] == "high"
        ]

        low_features = cluster_info[
            cluster_info["direction"] == "low"
        ]

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### 高於平均的特徵")

            st.dataframe(
                high_features[["feature", "z_mean"]],
                width="stretch"
            )

        with col2:

            st.markdown("#### 低於平均的特徵")

            st.dataframe(
                low_features[["feature", "z_mean"]],
                width="stretch"
            )

    else:

        st.info("找不到此 cluster 的解釋資料。")

else:

    st.info("尚未找到 soft_cluster_interpretation.csv。可先執行 cluster_interpretation_soft.py。")


# =========================
# Method Comparison
# =========================

st.subheader("推薦方法比較")

if method_comparison is not None:

    st.dataframe(
        method_comparison,
        width="stretch"
    )

    best_row = method_comparison.sort_values(
        "mean_similarity",
        ascending=False
    ).iloc[0]

    st.write(
        f"目前最佳方法：{best_row['method']}，"
        f"Mean Similarity = {best_row['mean_similarity']:.4f}，"
        f"Lift = {best_row['recommendation_lift']:.4f}"
    )

else:

    st.info("尚未找到 recommendation_method_comparison.csv。可先執行 evaluate_clusters.py。")


# =========================
# Figures
# =========================

st.subheader("分析圖表")

figure_paths = [
    ("推薦方法比較", "figures/recommendation_method_comparison.png"),
    ("Soft Cluster Heatmap", "figures/soft_cluster_feature_heatmap.png"),
    ("Soft Cluster Confidence", "figures/soft_cluster_confidence_distribution.png"),
    ("Correlation Matrix", "figures/correlation_matrix.png"),
]

for title, path in figure_paths:

    if os.path.exists(path):

        with st.expander(title):

            st.image(
                path,
                width="stretch"
            )


# =========================
# Footer
# =========================

st.divider()

st.markdown(
    """
### 系統流程

使用者輸入條件  
→ Soft Clustering  
→ 得到 Cluster 與 Confidence  
→ 使用 Cluster 平均行為特徵  
→ Cluster-based 景點推薦  
→ 推薦方法比較
"""
)