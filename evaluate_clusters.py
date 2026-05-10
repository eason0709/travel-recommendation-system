# 6

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# =========================
# 讀取資料
# =========================

user_df = pd.read_csv("data/travel_behavior_soft_clustered.csv")
attractions = pd.read_csv("data/attractions.csv")
interactions = pd.read_csv("data/user_interactions.csv")

# 重要：這裡要讀完整候選分數，而不是只讀 hybrid Top-K
recommendations = pd.read_csv("data/all_recommendation_candidates.csv")

# =========================
# DEBUG MODE
# =========================

DEBUG_MODE = True

if DEBUG_MODE:
    MAX_USERS = 2000
else:
    MAX_USERS = None

# =========================
# Debug subset
# =========================

if MAX_USERS is not None:

    selected_users = (
        user_df
        .sample(MAX_USERS, random_state=42)
        ["user_id"]
        .values
    )

    user_df = (
        user_df[
            user_df["user_id"].isin(selected_users)
        ]
        .reset_index(drop=True)
    )

    interactions = (
        interactions[
            interactions["user_id"].isin(selected_users)
        ]
        .reset_index(drop=True)
    )

    recommendations = (
        recommendations[
            recommendations["user_id"].isin(selected_users)
        ]
        .reset_index(drop=True)
    )

print()
print("DEBUG MODE:", DEBUG_MODE)
print("User count:", len(user_df))
print("Candidate recommendation count:", len(recommendations))

print()
print("名詞解釋")
print("-" * 60)
print("Personal-only：只用單一使用者的行為向量和景點向量相似度做推薦。")
print("Cluster-only：用使用者所屬 cluster 的平均行為向量做推薦。")
print("Collaborative-only：用同 cluster 使用者過去對景點的平均 rating 做推薦。")
print("Hybrid：融合 personal、cluster、collaborative 三種分數。")
print("Recommendation Lift：推薦結果相對 random baseline 的改善幅度。")
print("Random Baseline：隨機推薦景點，用來當最低比較基準。")
print("Mean Similarity：Top-K 推薦景點與使用者行為向量的平均相似度。")
print("-" * 60)


# =========================
# Feature 設定
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


def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# =========================
# 建立 user / attraction 向量查表
# =========================

user_behavior_map = {}

for _, row in user_df.iterrows():
    user_behavior_map[row["user_id"]] = row[behavior_features].values


attraction_map = {}

for _, row in attractions.iterrows():
    attraction_map[row["attraction_id"]] = row[attraction_features].values


# =========================
# 方法比較
# =========================

methods = {
    "personal_only": "personal_score",
    "cluster_only": "cluster_score",
    "collaborative_only": "collaborative_score",
    "hybrid": "hybrid_score"
}

rows = []

# =========================
# Random baseline
# =========================

np.random.seed(42)

random_scores = []

for user_id in user_df["user_id"].values:

    random_item = attractions.sample(
        1,
        random_state=int(user_id)
    ).iloc[0]

    sim = cosine_similarity(
        user_behavior_map[user_id],
        random_item[attraction_features].values
    )

    random_scores.append(sim)

random_baseline = np.mean(random_scores)

# =========================
# 各方法各自排序 Top-K
# =========================

top_k = 5

for method_name, score_col in methods.items():

    top_recs = (
        recommendations
        .sort_values(
            ["user_id", score_col],
            ascending=[True, False]
        )
        .groupby("user_id")
        .head(top_k)
    )

    sims = []

    for _, row in top_recs.iterrows():

        sim = cosine_similarity(
            user_behavior_map[row["user_id"]],
            attraction_map[row["attraction_id"]]
        )

        sims.append(sim)

    mean_sim = np.mean(sims)

    rows.append({
        "method": method_name,
        "mean_similarity": mean_sim,
        "random_baseline": random_baseline,
        "recommendation_lift": mean_sim - random_baseline
    })


eval_df = pd.DataFrame(rows)

print()
print("推薦方法比較")
print("-" * 60)
print(eval_df)

best = eval_df.sort_values(
    "mean_similarity",
    ascending=False
).iloc[0]

print()
print("最佳方法:")
print(best["method"])
print()
print("解讀:")

if best["method"] == "personal_only":
    print("個人行為向量最有預測力，代表使用者自身行為比群體資訊更重要。")
elif best["method"] == "cluster_only":
    print("cluster 平均行為最有預測力，代表分群對推薦有明顯價值。")
elif best["method"] == "collaborative_only":
    print("同群使用者的歷史互動最有預測力，代表 collaborative filtering 有效。")
else:
    print("混合方法最佳，代表個人、群體、協同訊號能互補。")


# =========================
# 儲存結果
# =========================

eval_df.to_csv(
    "data/recommendation_method_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================
# 視覺化
# =========================

plt.figure(figsize=(8, 5))

plt.bar(
    eval_df["method"],
    eval_df["mean_similarity"]
)

plt.axhline(
    random_baseline,
    linestyle="--",
    label="Random baseline"
)

plt.ylabel("Mean Similarity")
plt.title("Recommendation Method Comparison")
plt.xticks(rotation=25)
plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/recommendation_method_comparison.png",
    dpi=300
)

plt.show()

print()
print("Saved:")
print("data/recommendation_method_comparison.csv")
print("figures/recommendation_method_comparison.png")