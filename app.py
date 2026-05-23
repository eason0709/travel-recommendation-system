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
st.caption("Unsupervised Clustering + Personal / Cluster-based Recommendation Demo")


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
# Cluster Labels and Descriptions
# =========================

CLUSTER_LABELS = {
    0: "低預算室內文化型",
    1: "好天氣一般消費型",
    2: "極高預算特殊型",
    3: "自然景點導向型",
    4: "熱門社交打卡型",
    5: "高預算消費型",
    6: "壞天氣室內備案型",
    7: "拍照打卡型",
}

CLUSTER_DESCRIPTIONS = {
    0: "預算偏低，不太偏好自然景點，較偏向室內、文化或熱門景點。",
    1: "主要對應天氣較好的出遊情境，對熱門程度與拍照價值需求較低，稍微能接受較高消費。",
    2: "主要由極高預算特徵區分出來，人數可能較少，其他旅遊偏好不一定明確。",
    3: "明顯偏好自然景點，不太重視熱門程度、拍照價值或高消費景點。",
    4: "偏好熱門景點，社交情境與拍照價值也偏高，適合朋友出遊或社群分享型推薦。",
    5: "預算與可接受消費偏高，但不特別追求熱門、文化或美食價值。",
    6: "較常對應天氣較差的情境，且室內偏好較高，適合雨天備案或室內活動。",
    7: "最重視拍照價值，並且略偏好熱門與文化景點，適合視覺特色明顯的景點。",
}

FEATURE_DISPLAY_NAMES = {
    "budget": "預算",
    "available_time": "可用時間",
    "weather_badness": "天氣不佳程度",
    "social_context": "社交情境",
    "fatigue": "疲勞程度",
    "spontaneity": "隨興程度",
    "selected_indoor_score": "室內偏好",
    "selected_cost_level": "可接受消費",
    "selected_photo_value": "拍照價值",
    "selected_nature_value": "自然景點偏好",
    "selected_culture_value": "文化景點偏好",
    "selected_food_value": "美食偏好",
    "selected_popularity": "熱門程度偏好",
}


# 這些名稱不是外部真實景點資料，而是讓模擬景點能對應到較接近現實的場景名稱。
# 原始 Attraction_XXX 仍保留作為內部 ID，避免和既有資料流程衝突。
REALISTIC_SCENE_NAME_POOLS = {
    "museum": [
        "城市歷史博物館", "河岸美術館", "自然科學展示館", "地方文化博物館", "當代藝術展覽館",
    ],
    "nature_trail": [
        "森林步道", "海岸觀景步道", "山區生態步道", "湖畔自然步道", "溪谷健行路線",
    ],
    "night_market": [
        "在地夜市", "河岸觀光夜市", "老城小吃夜市", "車站商圈夜市", "廟口美食夜市",
    ],
    "cafe_street": [
        "文青咖啡街", "老屋咖啡巷", "河岸咖啡街區", "文創甜點街", "巷弄咖啡聚落",
    ],
    "historic_area": [
        "老街歷史街區", "古城文化街區", "傳統聚落保存區", "港町歷史街區", "紅磚建築街區",
    ],
    "shopping_mall": [
        "大型購物中心", "車站百貨商圈", "室內娛樂商場", "複合式生活商場", "都會購物廣場",
    ],
    "park": [
        "都會公園", "河濱公園", "森林公園", "親子休閒公園", "湖畔景觀公園",
    ],
    "temple": [
        "百年古廟", "山城寺廟", "廟口文化廣場", "傳統信仰中心", "歷史寺院",
    ],
}


def infer_primary_attraction_type(attraction_type):

    if pd.isna(attraction_type):
        return "unknown"

    return str(attraction_type).split("+")[0]


def build_realistic_scene_name(row):

    if "realistic_scene_name" in row.index and pd.notna(row["realistic_scene_name"]):
        return str(row["realistic_scene_name"])

    attraction_type = row.get("attraction_type", "unknown")
    primary_type = infer_primary_attraction_type(attraction_type)
    pool = REALISTIC_SCENE_NAME_POOLS.get(primary_type, ["旅遊景點"])

    try:
        attraction_id = int(row.get("attraction_id", row.name))
    except Exception:
        attraction_id = int(row.name)

    base_name = pool[attraction_id % len(pool)]

    if "+" in str(attraction_type):
        return f"{base_name} 複合景點 {attraction_id:03d}"

    return f"{base_name} {attraction_id:03d}"


def get_confidence_level(confidence):

    confidence = float(confidence)

    if confidence >= 0.70:
        return (
            "高信心",
            "模型對此使用者所屬 cluster 的判定較明確，可主要依據目前 cluster 解讀推薦結果。"
        )

    if confidence >= 0.40:
        return (
            "中等信心",
            "模型大致能判定使用者偏好，但仍建議參考 Soft Cluster Probability 中排名較高的其他 cluster。"
        )

    return (
        "低信心",
        "模型對單一 cluster 的判定不夠明確，表示使用者偏好可能分散於多個 cluster。建議不要只依賴目前 hard cluster 解釋。"
    )


def get_cluster_label(cluster_id):

    return CLUSTER_LABELS.get(
        int(cluster_id),
        f"Cluster {int(cluster_id)}"
    )


def get_cluster_description(cluster_id):

    return CLUSTER_DESCRIPTIONS.get(
        int(cluster_id),
        "目前尚未定義此 cluster 的人工語意解釋。"
    )


def format_cluster_title(cluster_id):

    return f"Cluster {int(cluster_id)}：{get_cluster_label(cluster_id)}"


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


def build_recommendation_row(spot, recommendation_score, personal_score=None, cluster_score=None):

    row = {
        "scene_name": build_realistic_scene_name(spot),
        "name": spot["name"],
        "attraction_type": spot["attraction_type"],
        "recommendation_score": recommendation_score,
        "cost_level": spot["cost_level"],
        "indoor_score": spot["indoor_score"],
        "photo_value": spot["photo_value"],
        "nature_value": spot["nature_value"],
        "culture_value": spot["culture_value"],
        "food_value": spot["food_value"],
        "popularity": spot["popularity"],
        "estimated_time": spot["estimated_time"]
    }

    if personal_score is not None:
        row["personal_score"] = personal_score

    if cluster_score is not None:
        row["cluster_score"] = cluster_score

    return row


def sort_top_k(rows, top_k):

    result = pd.DataFrame(rows)

    return (
        result
        .sort_values(
            "recommendation_score",
            ascending=False
        )
        .head(top_k)
    )


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
        score = cosine_similarity(cluster_vec, spot_vec)

        rows.append(
            build_recommendation_row(
                spot=spot,
                recommendation_score=score,
                cluster_score=score
            )
        )

    return sort_top_k(rows, top_k)


def get_personal_only_recommendations(
    selected_user,
    top_k,
    attractions_df
):

    user_vec = selected_user[behavior_features].values.astype(float)

    rows = []

    for _, spot in attractions_df.iterrows():

        spot_vec = spot[attraction_features].values.astype(float)
        score = cosine_similarity(user_vec, spot_vec)

        rows.append(
            build_recommendation_row(
                spot=spot,
                recommendation_score=score,
                personal_score=score
            )
        )

    return sort_top_k(rows, top_k)


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



# 若 attractions.csv 是舊版資料，沒有 realistic_scene_name 欄位，App 會自動補上較接近現實場景的顯示名稱。
if "realistic_scene_name" not in attractions.columns:
    attractions["realistic_scene_name"] = attractions.apply(
        build_realistic_scene_name,
        axis=1
    )

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

recommendation_mode = st.sidebar.radio(
    "Recommendation Mode",
    options=["personal_only", "cluster_only"],
    index=0,
    help=(
        "personal_only 直接使用個人行為向量與景點特徵計算相似度；"
        "cluster_only 使用所屬 cluster 的平均行為特徵與景點特徵計算相似度。"
    )
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

current_cluster_id = int(selected_user["soft_cluster"])
current_cluster_label = get_cluster_label(current_cluster_id)
current_cluster_description = get_cluster_description(current_cluster_id)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Soft Cluster",
        current_cluster_id
    )

with col2:
    st.metric(
        "Cluster 類型",
        current_cluster_label
    )

with col3:
    st.metric(
        "Cluster Confidence",
        f"{selected_user['soft_cluster_confidence']:.3f}"
    )

with col4:
    st.metric(
        "Budget",
        f"{selected_user['budget']:.1f}"
    )

st.markdown("### 決策系統判定結果")
st.info(
    f"目前系統判定此使用者屬於 **{format_cluster_title(current_cluster_id)}**。"
    f"{current_cluster_description}"
)

cluster_summary_df = pd.DataFrame([
    {
        "cluster": current_cluster_id,
        "cluster_label": current_cluster_label,
        "confidence": float(selected_user["soft_cluster_confidence"]),
        "description": current_cluster_description,
    }
])

st.dataframe(
    cluster_summary_df,
    width="stretch",
    hide_index=True
)

confidence_level, confidence_message = get_confidence_level(
    selected_user["soft_cluster_confidence"]
)

st.markdown("### 模型信心判讀")

confidence_df = pd.DataFrame([
    {
        "confidence_level": confidence_level,
        "confidence": float(selected_user["soft_cluster_confidence"]),
        "interpretation": confidence_message,
    }
])

st.dataframe(
    confidence_df,
    width="stretch",
    hide_index=True
)

if confidence_level == "低信心":
    st.warning(
        "此使用者的 cluster confidence 較低。建議同時查看 Soft Cluster Probability 中的前幾個 cluster，"
        "不要只把推薦結果解釋為單一 cluster 的偏好。"
    )
elif confidence_level == "中等信心":
    st.info(
        "此使用者仍可能同時具有其他 cluster 的偏好特徵。"
    )

st.markdown("### Cluster 計算流程說明")

with st.expander("查看模型如何計算 Soft Cluster", expanded=False):

    st.write(
        "本系統的 cluster 不是人工規則直接分類，而是由 Gaussian Mixture Model 根據使用者行為特徵計算機率後決定。"
    )

    st.markdown(
        """
計算流程如下：

1. 讀取使用者輸入特徵，例如 budget、available_time、weather_badness、fatigue、selected_indoor_score 等。
2. 使用訓練時儲存的 imputer 補齊缺失值。
3. 使用訓練時儲存的 StandardScaler 將不同尺度的欄位轉成標準化數值。
4. 將標準化後的特徵輸入 GMM。
5. GMM 計算使用者屬於每個 cluster 的 probability。
6. probability 最高的 cluster 會被顯示為 Soft Cluster。
7. 最高 probability 會被視為 Cluster Confidence。

因此，畫面上的 cluster label 是模型輸出的機率結果，而不是單純用 if-else 規則指定。
        """
    )

    st.write("模型實際使用的特徵：")

    model_features_df = pd.DataFrame({
        "feature": cluster_model["features"],
        "feature_name": [
            FEATURE_DISPLAY_NAMES.get(feature, feature)
            for feature in cluster_model["features"]
        ]
    })

    st.dataframe(
        model_features_df,
        width="stretch",
        hide_index=True
    )

    if manual_mode and cluster_probs is not None:
        top_prob_df = pd.DataFrame({
            "cluster": list(range(len(cluster_probs))),
            "cluster_label": [
                get_cluster_label(cluster_id)
                for cluster_id in range(len(cluster_probs))
            ],
            "probability": cluster_probs
        }).sort_values("probability", ascending=False).head(3)

        st.write("目前輸入最可能屬於的前三個 cluster：")
        st.dataframe(
            top_prob_df,
            width="stretch",
            hide_index=True
        )


if manual_mode and cluster_probs is not None:

    st.markdown("### Soft Cluster Probability")

    prob_df = pd.DataFrame({
        "cluster": list(range(len(cluster_probs))),
        "cluster_label": [
            get_cluster_label(cluster_id)
            for cluster_id in range(len(cluster_probs))
        ],
        "probability": cluster_probs
    })

    prob_df = prob_df.sort_values(
        "probability",
        ascending=False
    )

    st.dataframe(
        prob_df,
        width="stretch",
        hide_index=True
    )

    st.bar_chart(
        prob_df.set_index("cluster_label")["probability"]
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
# Recommendation
# =========================

st.subheader("推薦景點")

cluster_id = int(selected_user["soft_cluster"])

if cluster_id not in cluster_behavior.index:

    st.warning("目前 cluster 不存在於 cluster_behavior 中，請重新確認模型與資料是否一致。")
    st.stop()

if recommendation_mode == "personal_only":

    st.info(
        "目前使用 personal_only 推薦：直接比較使用者個人行為特徵向量與景點特徵向量。"
        "這個模式通常會有較高的個人化相似度，但 cluster 解釋主要作為輔助說明。"
    )

    user_recs = get_personal_only_recommendations(
        selected_user=selected_user,
        top_k=top_k,
        attractions_df=attractions
    )

else:

    st.info(
        "目前使用 cluster_only 推薦：比較使用者所屬 cluster 的平均行為特徵與景點特徵。"
        "這個模式較適合展示分群語意與冷啟動情境，但個人化程度可能低於 personal_only。"
    )

    user_recs = get_cluster_only_recommendations(
        cluster_id=cluster_id,
        top_k=top_k,
        cluster_behavior=cluster_behavior,
        attractions_df=attractions
    )

if len(user_recs) == 0:

    st.warning("目前沒有推薦結果。")

else:

    st.caption(f"目前 Recommendation Mode：{recommendation_mode}")

    display_cols = [
        "scene_name",
        "name",
        "attraction_type",
        "recommendation_score",
        "personal_score",
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

    st.caption(
        "scene_name 是為了展示用途產生的現實場景式名稱；name 保留為系統內部產生的景點代碼。"
    )

    st.markdown("### 推薦結果解釋")

    for _, row in user_recs.iterrows():

        with st.expander(f"{row.get('scene_name', row['name'])}"):

            st.write(f"場景名稱：{row.get('scene_name', row['name'])}")
            st.write(f"內部名稱：{row['name']}")
            st.write(f"景點類型：{row.get('attraction_type', 'N/A')}")
            st.write(f"Recommendation Score：{row['recommendation_score']:.4f}")

            if recommendation_mode == "personal_only":
                st.write(f"Personal Score：{row['personal_score']:.4f}")
                st.write(
                    "解釋：此推薦是將使用者目前輸入或既有資料中的個人行為特徵向量，"
                    "直接與景點特徵向量進行相似度比較。"
                )
            else:
                st.write(f"Cluster Score：{row['cluster_score']:.4f}")
                st.write(
                    "解釋：此推薦是將景點特徵向量與該 cluster 的平均行為特徵向量進行相似度比較。"
                )


# =========================
# Cluster Interpretation
# =========================

st.subheader("Cluster 解釋")

cluster_id = int(selected_user["soft_cluster"])

st.markdown(f"### {format_cluster_title(cluster_id)}")
st.write(get_cluster_description(cluster_id))

if cluster_interpretation is not None:

    cluster_info = cluster_interpretation[
        cluster_interpretation["cluster"] == cluster_id
    ].copy()

    if len(cluster_info) > 0:

        cluster_info["feature_name"] = cluster_info["feature"].map(
            FEATURE_DISPLAY_NAMES
        ).fillna(cluster_info["feature"])

        high_features = cluster_info[
            cluster_info["direction"] == "high"
        ].sort_values("z_mean", ascending=False)

        low_features = cluster_info[
            cluster_info["direction"] == "low"
        ].sort_values("z_mean", ascending=True)

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### 高於平均的特徵")

            st.dataframe(
                high_features[["feature_name", "feature", "z_mean"]],
                width="stretch",
                hide_index=True
            )

        with col2:

            st.markdown("#### 低於平均的特徵")

            st.dataframe(
                low_features[["feature_name", "feature", "z_mean"]],
                width="stretch",
                hide_index=True
            )

        st.caption(
            "z_mean 是標準化後的群體平均值；正值代表相對整體偏高，負值代表相對整體偏低。"
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
→ 得到 Cluster、Confidence 與 Probability  
→ 判讀模型信心與可能的不確定性  
→ 選擇 Recommendation Mode  
→ personal_only 或 cluster_only 景點推薦  
→ 推薦方法比較
"""
)