# 3

import os
import numpy as np
import pandas as pd


np.random.seed(42)

os.makedirs("data", exist_ok=True)

num_attractions = 300
records = []


def clip_score(x):
    return float(np.clip(x, 0, 5))


attraction_types = {
    "museum": {
        "indoor_score": 4.6,
        "cost_level": 2.8,
        "photo_value": 3.8,
        "nature_value": 0.5,
        "culture_value": 4.8,
        "food_value": 1.2,
        "popularity": 3.2,
        "estimated_time": 2.5
    },
    "nature_trail": {
        "indoor_score": 0.4,
        "cost_level": 1.2,
        "photo_value": 4.0,
        "nature_value": 4.8,
        "culture_value": 0.8,
        "food_value": 1.0,
        "popularity": 2.8,
        "estimated_time": 4.5
    },
    "night_market": {
        "indoor_score": 1.8,
        "cost_level": 2.0,
        "photo_value": 3.0,
        "nature_value": 0.3,
        "culture_value": 2.5,
        "food_value": 4.8,
        "popularity": 4.4,
        "estimated_time": 2.2
    },
    "cafe_street": {
        "indoor_score": 3.3,
        "cost_level": 3.0,
        "photo_value": 4.3,
        "nature_value": 0.8,
        "culture_value": 2.5,
        "food_value": 4.0,
        "popularity": 3.8,
        "estimated_time": 2.0
    },
    "historic_area": {
        "indoor_score": 2.2,
        "cost_level": 1.8,
        "photo_value": 3.8,
        "nature_value": 1.2,
        "culture_value": 4.5,
        "food_value": 2.0,
        "popularity": 3.2,
        "estimated_time": 3.0
    },
    "shopping_mall": {
        "indoor_score": 4.8,
        "cost_level": 4.0,
        "photo_value": 2.8,
        "nature_value": 0.2,
        "culture_value": 1.8,
        "food_value": 3.8,
        "popularity": 4.3,
        "estimated_time": 3.2
    },
    "park": {
        "indoor_score": 0.5,
        "cost_level": 0.8,
        "photo_value": 3.5,
        "nature_value": 4.2,
        "culture_value": 1.0,
        "food_value": 1.3,
        "popularity": 2.5,
        "estimated_time": 2.8
    },
    "temple": {
        "indoor_score": 2.5,
        "cost_level": 1.0,
        "photo_value": 3.4,
        "nature_value": 1.2,
        "culture_value": 4.6,
        "food_value": 1.5,
        "popularity": 2.8,
        "estimated_time": 2.0
    }
}

type_names = list(attraction_types.keys())
type_probs = [0.12, 0.13, 0.14, 0.14, 0.13, 0.12, 0.12, 0.10]

for attraction_id in range(1, num_attractions + 1):

    primary_type = np.random.choice(type_names, p=type_probs)
    base = attraction_types[primary_type].copy()

    # 30% 景點是混合型，例如 cafe + culture、nature + photo
    if np.random.rand() < 0.30:
        secondary_type = np.random.choice(type_names)
        second = attraction_types[secondary_type]

        mix = np.random.uniform(0.20, 0.45)

        for key in base:
            base[key] = base[key] * (1 - mix) + second[key] * mix

        attraction_type = f"{primary_type}+{secondary_type}"
    else:
        attraction_type = primary_type

    # 加入景點個體差異
    indoor_score = clip_score(base["indoor_score"] + np.random.normal(0, 0.55))
    cost_level = clip_score(base["cost_level"] + np.random.normal(0, 0.65))
    photo_value = clip_score(base["photo_value"] + np.random.normal(0, 0.65))
    nature_value = clip_score(base["nature_value"] + np.random.normal(0, 0.65))
    culture_value = clip_score(base["culture_value"] + np.random.normal(0, 0.65))
    food_value = clip_score(base["food_value"] + np.random.normal(0, 0.65))
    popularity = clip_score(base["popularity"] + np.random.normal(0, 0.80))

    estimated_time = float(
        np.clip(
            base["estimated_time"] + np.random.normal(0, 0.8),
            0.8,
            8.0
        )
    )

    # 長尾偏誤：少數景點非常熱門，多數普通
    if np.random.rand() < 0.08:
        popularity = clip_score(popularity + np.random.uniform(1.0, 2.0))

    if np.random.rand() < 0.08:
        popularity = clip_score(popularity - np.random.uniform(1.0, 2.0))

    name = f"Attraction_{attraction_id:03d}_{primary_type}"

    records.append({
        "attraction_id": attraction_id,
        "name": name,
        "attraction_type": attraction_type,
        "cost_level": cost_level,
        "indoor_score": indoor_score,
        "photo_value": photo_value,
        "nature_value": nature_value,
        "culture_value": culture_value,
        "food_value": food_value,
        "popularity": popularity,
        "estimated_time": estimated_time
    })


df = pd.DataFrame(records)

df.to_csv(
    "data/attractions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("data/attractions.csv 已產生")
print(df.head())
print()
print("Dataset shape:")
print(df.shape)
print()
print("Attraction type distribution:")
print(df["attraction_type"].value_counts().head(20))