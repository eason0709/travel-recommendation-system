# 4

import os
import numpy as np
import pandas as pd


np.random.seed(42)

os.makedirs("data", exist_ok=True)

# =========================
# 可調參數
# =========================

DEBUG_MODE = True

if DEBUG_MODE:

    MAX_USERS = 2000
    MAX_ATTRACTIONS = 50

else:

    MAX_USERS = None
    MAX_ATTRACTIONS = None

user_df = pd.read_csv("data/travel_behavior_soft_clustered.csv")
attractions = pd.read_csv("data/attractions.csv")

# =========================
# Debug subset
# =========================

if MAX_USERS is not None:
    user_df = user_df.sample(
        MAX_USERS,
        random_state=42
    ).reset_index(drop=True)

if MAX_ATTRACTIONS is not None:
    attractions = attractions.sample(
        MAX_ATTRACTIONS,
        random_state=42
    ).reset_index(drop=True)

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


def softmax(x, temperature=0.45):
    x = np.array(x, dtype=float)
    x = x / temperature
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


interaction_rows = []

for idx_user, (_, user) in enumerate(user_df.iterrows()):

    # =========================
    # Progress Print
    # =========================

    if idx_user % 200 == 0:

        print(
            f"Processing user "
            f"{idx_user} / {len(user_df)}"
        )

    user_vec = user[behavior_features].values

    scores = []

    for _, spot in attractions.iterrows():

        spot_vec = spot[attraction_features].values

        similarity = cosine_similarity(user_vec, spot_vec)

        time_penalty = max(
            0,
            spot["estimated_time"] - user["available_time"]
        ) * 0.08

        weather_penalty = (
            user["weather_badness"]
            * max(0, 2.5 - spot["indoor_score"])
            * 0.10
        )

        popularity_bias = spot["popularity"] * 0.04

        score = (
            similarity * 1.2
            - time_penalty
            - weather_penalty
            + popularity_bias
            + np.random.normal(0, 0.05)
        )

        scores.append(score)

    probabilities = softmax(scores, temperature=0.45)

    # 每個使用者模擬 5 次歷史互動
    chosen_indices = np.random.choice(
        attractions.index,
        size=5,
        replace=False,
        p=probabilities
    )

    for idx in chosen_indices:

        spot = attractions.loc[idx]
        spot_vec = spot[attraction_features].values

        similarity = cosine_similarity(user_vec, spot_vec)

        # rating 是隱含回饋 proxy
        rating = np.clip(
            2.5
            + similarity * 2.0
            - max(0, spot["estimated_time"] - user["available_time"]) * 0.15
            - user["weather_badness"] * max(0, 2.5 - spot["indoor_score"]) * 0.15
            + np.random.normal(0, 0.45),
            1,
            5
        )

        clicked = int(rating >= 4.0)

        interaction_rows.append({
            "user_id": user["user_id"],
            "soft_cluster": user["soft_cluster"],
            "attraction_id": spot["attraction_id"],
            "rating": rating,
            "clicked": clicked
        })


interactions = pd.DataFrame(interaction_rows)

interactions.to_csv(
    "data/user_interactions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("data/user_interactions.csv 已產生")
print(interactions.head())
print()
print("Dataset shape:")
print(interactions.shape)
print()
print("Rating summary:")
print(interactions["rating"].describe())
print()
print("Clicked distribution:")
print(interactions["clicked"].value_counts())