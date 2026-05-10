# 1

import os
import numpy as np
import pandas as pd


np.random.seed(42)

os.makedirs("data", exist_ok=True)

num_users = 20000
records = []


def clip_score(x):
    return float(np.clip(x, 0, 5))


def clip_01(x):
    return float(np.clip(x, 0, 1))


groups = {
    "food":      [1.6, 1.5, 2.2, 4.5, 3.0],
    "photo":    [4.5, 2.3, 3.4, 2.3, 3.8],
    "nature":   [2.0, 4.5, 1.8, 2.2, 2.0],
    "culture":  [3.0, 2.0, 4.5, 2.4, 2.8],
    "trend":    [4.0, 2.0, 3.0, 3.0, 4.6],
    "balanced": [2.8, 2.8, 2.8, 2.8, 2.8],
}

group_names = list(groups.keys())
group_probs = [0.18, 0.16, 0.16, 0.16, 0.14, 0.20]


for user_id in range(1, num_users + 1):

    # =========================
    # 內部 latent group
    # 不輸出到 CSV
    # =========================

    primary_group = np.random.choice(group_names, p=group_probs)
    center = np.array(groups[primary_group], dtype=float)

    # 35% 使用者是混合型，不是單一族群
    if np.random.rand() < 0.35:
        secondary_group = np.random.choice(group_names)
        second_center = np.array(groups[secondary_group], dtype=float)
        mix_ratio = np.random.uniform(0.20, 0.45)
        center = center * (1 - mix_ratio) + second_center * mix_ratio

    # =========================
    # Latent traits
    # =========================

    photo_pref = clip_score(np.random.normal(center[0], 0.85))
    nature_pref = clip_score(np.random.normal(center[1], 0.85))
    culture_pref = clip_score(np.random.normal(center[2], 0.85))
    food_pref = clip_score(np.random.normal(center[3], 0.85))
    popularity_pref = clip_score(np.random.normal(center[4], 0.85))

    adventure = (nature_pref - 2.5) / 1.5 + np.random.normal(0, 0.4)
    social = (food_pref + popularity_pref - 5) / 3 + np.random.normal(0, 0.4)
    luxury = np.random.normal(0, 1)
    comfort = np.random.normal(0, 1)
    trendiness = (popularity_pref - 2.5) / 1.5 + np.random.normal(0, 0.4)

    # =========================
    # 基本條件
    # =========================

    budget = np.clip(
        np.random.lognormal(
            mean=6.45 + luxury * 0.22,
            sigma=0.50
        ),
        100,
        5000
    )

    fatigue = clip_score(
        2.5
        + comfort * 0.35
        - adventure * 0.15
        + np.random.normal(0, 1.0)
    )

    spontaneity = clip_score(
        2.5
        + adventure * 0.45
        + trendiness * 0.20
        - comfort * 0.20
        + np.random.normal(0, 0.95)
    )

    available_time = np.clip(
        np.random.normal(
            5
            + adventure * 0.35
            + spontaneity * 0.15
            - fatigue * 0.18,
            1.6
        ),
        1,
        12
    )

    weather_badness = clip_01(
        np.random.beta(2, 5)
        + np.random.normal(0, 0.06)
    )

    social_context = clip_01(
        0.5
        + social * 0.16
        + np.random.normal(0, 0.18)
    )

    # =========================
    # 情境壓力
    # =========================

    fatigue_pressure = fatigue / 5
    budget_pressure = np.clip((850 - budget) / 850, 0, 1)
    spontaneity_level = spontaneity / 5

    # =========================
    # 行為生成
    # =========================

    selected_indoor_score = clip_score(
        2.0
        + weather_badness * 0.75
        + fatigue_pressure * 0.55
        + comfort * 0.18
        - adventure * 0.18
        + culture_pref * 0.10
        + np.random.normal(0, 1.05)
    )

    selected_cost_level = clip_score(
        2.5
        + luxury * 0.48
        - budget_pressure * 0.55
        + social_context * 0.22
        + np.random.normal(0, 1.05)
    )

    selected_photo_value = clip_score(
        1.3
        + photo_pref * 0.48
        + popularity_pref * 0.18
        + social_context * 0.18
        + spontaneity_level * 0.25
        - fatigue_pressure * 0.18
        + np.random.normal(0, 1.05)
    )

    selected_nature_value = clip_score(
        1.3
        + nature_pref * 0.50
        + adventure * 0.20
        - weather_badness * 0.50
        - fatigue_pressure * 0.22
        + spontaneity_level * 0.22
        + np.random.normal(0, 1.05)
    )

    selected_culture_value = clip_score(
        1.3
        + culture_pref * 0.50
        + photo_pref * 0.10
        + selected_indoor_score * 0.08
        + np.random.normal(0, 1.05)
    )

    selected_food_value = clip_score(
        1.3
        + food_pref * 0.50
        + social_context * 0.30
        + comfort * 0.12
        + np.random.normal(0, 1.05)
    )

    selected_popularity = clip_score(
        1.3
        + popularity_pref * 0.50
        + trendiness * 0.22
        + social_context * 0.25
        + photo_pref * 0.08
        + np.random.normal(0, 1.05)
    )

    # =========================
    # 矛盾 / 非理性使用者
    # =========================

    if np.random.rand() < 0.12:
        selected_indoor_score = clip_score(selected_indoor_score + np.random.normal(0, 1.2))
        selected_photo_value = clip_score(selected_photo_value + np.random.normal(0, 1.4))
        selected_nature_value = clip_score(selected_nature_value + np.random.normal(0, 1.4))
        selected_culture_value = clip_score(selected_culture_value + np.random.normal(0, 1.4))
        selected_food_value = clip_score(selected_food_value + np.random.normal(0, 1.4))
        selected_popularity = clip_score(selected_popularity + np.random.normal(0, 1.4))

    # =========================
    # Outlier
    # =========================

    if np.random.rand() < 0.025:
        budget = np.random.choice([100, 5000])
        selected_indoor_score = np.random.uniform(0, 5)
        selected_cost_level = np.random.uniform(0, 5)
        selected_photo_value = np.random.uniform(0, 5)
        selected_nature_value = np.random.uniform(0, 5)
        selected_culture_value = np.random.uniform(0, 5)
        selected_food_value = np.random.uniform(0, 5)
        selected_popularity = np.random.uniform(0, 5)

    # =========================
    # Missing value
    # =========================

    if np.random.rand() < 0.05:
        photo_pref = np.nan
    if np.random.rand() < 0.05:
        culture_pref = np.nan
    if np.random.rand() < 0.05:
        food_pref = np.nan

    records.append({
        "user_id": user_id,

        "budget": budget,
        "available_time": available_time,
        "weather_badness": weather_badness,
        "social_context": social_context,
        "fatigue": fatigue,
        "spontaneity": spontaneity,

        "selected_indoor_score": selected_indoor_score,
        "selected_cost_level": selected_cost_level,
        "selected_photo_value": selected_photo_value,
        "selected_nature_value": selected_nature_value,
        "selected_culture_value": selected_culture_value,
        "selected_food_value": selected_food_value,
        "selected_popularity": selected_popularity
    })


df = pd.DataFrame(records)

df.to_csv(
    "data/travel_behavior.csv",
    index=False,
    encoding="utf-8-sig"
)

print("data/travel_behavior.csv 已產生")
print(df.head())
print()
print("Dataset shape:", df.shape)
print()
print("Missing values:")
print(df.isnull().sum())