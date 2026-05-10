# 5

import os
import numpy as np
import pandas as pd


os.makedirs("data", exist_ok=True)

# =========================
# DEBUG MODE
# =========================

DEBUG_MODE = True

if DEBUG_MODE:

    MAX_USERS = 2000
    MAX_ATTRACTIONS = 50

else:

    MAX_USERS = None
    MAX_ATTRACTIONS = None

# =========================
# 讀取資料
# =========================

user_df = pd.read_csv("data/travel_behavior_soft_clustered.csv")
attractions = pd.read_csv("data/attractions.csv")
interactions = pd.read_csv("data/user_interactions.csv")

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

if MAX_ATTRACTIONS is not None:

    selected_attractions = (
        attractions
        .sample(MAX_ATTRACTIONS, random_state=42)
        ["attraction_id"]
        .values
    )

    attractions = (
        attractions[
            attractions["attraction_id"].isin(selected_attractions)
        ]
        .reset_index(drop=True)
    )

    interactions = (
        interactions[
            interactions["attraction_id"].isin(selected_attractions)
        ]
        .reset_index(drop=True)
    )

print()
print("DEBUG MODE:", DEBUG_MODE)
print("User count:", len(user_df))
print("Attraction count:", len(attractions))
print("Interaction count:", len(interactions))

numeric_cols = user_df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    user_df[col] = user_df[col].fillna(user_df[col].mean())


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


def user_behavior_vector(user):
    return user[behavior_features].values.astype(float)


def attraction_vector(spot):
    return spot[attraction_features].values.astype(float)


cluster_behavior = (
    user_df
    .groupby("soft_cluster")[behavior_features]
    .mean()
)

cluster_item_rating = (
    interactions
    .groupby(["soft_cluster", "attraction_id"])["rating"]
    .mean()
    .reset_index()
)

global_item_rating = (
    interactions
    .groupby("attraction_id")["rating"]
    .mean()
    .to_dict()
)

cluster_item_rating_map = {}

for _, row in cluster_item_rating.iterrows():
    cluster_item_rating_map[
        (row["soft_cluster"], row["attraction_id"])
    ] = row["rating"]


def personal_score(user, spot):
    sim = cosine_similarity(
        user_behavior_vector(user),
        attraction_vector(spot)
    )

    time_penalty = max(
        0,
        spot["estimated_time"] - user["available_time"]
    ) * 0.08

    weather_penalty = (
        user["weather_badness"]
        * max(0, 2.5 - spot["indoor_score"])
        * 0.10
    )

    return sim - time_penalty - weather_penalty


def cluster_only_score(user, spot):
    cluster_id = user["soft_cluster"]

    cluster_vec = cluster_behavior.loc[cluster_id].values

    return cosine_similarity(
        cluster_vec,
        attraction_vector(spot)
    )


def collaborative_score(user, spot):
    cluster_id = user["soft_cluster"]
    attraction_id = spot["attraction_id"]

    cluster_rating = cluster_item_rating_map.get(
        (cluster_id, attraction_id),
        np.nan
    )

    if np.isnan(cluster_rating):
        cluster_rating = global_item_rating.get(attraction_id, 3.0)

    # rating 1~5 轉成約 0~1
    return (cluster_rating - 1) / 4


def hybrid_score(user, spot):
    p = personal_score(user, spot)
    c = cluster_only_score(user, spot)
    cf = collaborative_score(user, spot)

    confidence = user["soft_cluster_confidence"]

    personal_weight = 0.55
    cluster_weight = 0.20 * confidence
    collaborative_weight = 0.25 * confidence

    total = personal_weight + cluster_weight + collaborative_weight

    return (
        personal_weight * p
        + cluster_weight * c
        + collaborative_weight * cf
    ) / total


top_k = 5

all_candidates = []
recommendation_rows = []

for idx_user, (_, user) in enumerate(user_df.iterrows()):

    if idx_user % 100 == 0:

        print(
            f"Processing user "
            f"{idx_user} / {len(user_df)}"
        )

    rows = []

    seen_items = set(
        interactions.loc[
            interactions["user_id"] == user["user_id"],
            "attraction_id"
        ].values
    )

    for _, spot in attractions.iterrows():

        # 推薦未看過的景點
        if spot["attraction_id"] in seen_items:
            continue

        p_score = personal_score(user, spot)
        c_score = cluster_only_score(user, spot)
        cf_score = collaborative_score(user, spot)
        h_score = hybrid_score(user, spot)

        rows.append({
            "user_id": user["user_id"],
            "soft_cluster": user["soft_cluster"],
            "cluster_confidence": user["soft_cluster_confidence"],
            "attraction_id": spot["attraction_id"],
            "name": spot["name"],
            "attraction_type": spot["attraction_type"],
            "personal_score": p_score,
            "cluster_score": c_score,
            "collaborative_score": cf_score,
            "hybrid_score": h_score,
            "recommendation_score": h_score
        })

    scored = pd.DataFrame(rows)

    all_candidates.append(scored)

    top = (
        scored
        .sort_values("recommendation_score", ascending=False)
        .head(top_k)
    )

    recommendation_rows.append(top)


all_candidates_df = pd.concat(
    all_candidates,
    ignore_index=True
)

recommendations = pd.concat(
    recommendation_rows,
    ignore_index=True
)

all_candidates_df.to_csv(
    "data/all_recommendation_candidates.csv",
    index=False,
    encoding="utf-8-sig"
)

recommendations.to_csv(
    "data/recommendations.csv",
    index=False,
    encoding="utf-8-sig"
)

print("data/all_recommendation_candidates.csv 已產生")
print("data/recommendations.csv 已產生")
print(recommendations.head(20))