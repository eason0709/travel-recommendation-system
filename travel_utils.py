"""Shared utilities for the travel recommendation project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RANDOM_STATE = 42
DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")
MODEL_DIR = Path("models")

BEHAVIOR_FEATURES = [
    "selected_indoor_score",
    "selected_cost_level",
    "selected_photo_value",
    "selected_nature_value",
    "selected_culture_value",
    "selected_food_value",
    "selected_popularity",
]

ATTRACTION_FEATURES = [
    "indoor_score",
    "cost_level",
    "photo_value",
    "nature_value",
    "culture_value",
    "food_value",
    "popularity",
]

MODEL_FEATURES = [
    "budget",
    "available_time",
    "weather_badness",
    "social_context",
    "fatigue",
    "spontaneity",
    *BEHAVIOR_FEATURES,
]

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

ATTRACTION_PROFILES = {
    "museum": [4.6, 2.8, 3.8, 0.5, 4.8, 1.2, 3.2, 2.5],
    "nature_trail": [0.4, 1.2, 4.0, 4.8, 0.8, 1.0, 2.8, 4.5],
    "night_market": [1.8, 2.0, 3.0, 0.3, 2.5, 4.8, 4.4, 2.2],
    "cafe_street": [3.3, 3.0, 4.3, 0.8, 2.5, 4.0, 3.8, 2.0],
    "historic_area": [2.2, 1.8, 3.8, 1.2, 4.5, 2.0, 3.2, 3.0],
    "shopping_mall": [4.8, 4.0, 2.8, 0.2, 1.8, 3.8, 4.3, 3.2],
    "park": [0.5, 0.8, 3.5, 4.2, 1.0, 1.3, 2.5, 2.8],
    "temple": [2.5, 1.0, 3.4, 1.2, 4.6, 1.5, 2.8, 2.0],
}

ATTRACTION_TYPES = list(ATTRACTION_PROFILES)
ATTRACTION_TYPE_PROBS = np.array([0.12, 0.13, 0.14, 0.14, 0.13, 0.12, 0.12, 0.10])
ATTRACTION_PROFILE_COLUMNS = [*ATTRACTION_FEATURES, "estimated_time"]

REALISTIC_SCENE_NAME_POOLS = {
    "museum": ["城市歷史博物館", "河岸美術館", "自然科學展示館", "地方文化博物館", "當代藝術展覽館"],
    "nature_trail": ["森林步道", "海岸觀景步道", "山區生態步道", "湖畔自然步道", "溪谷健行路線"],
    "night_market": ["在地夜市", "河岸觀光夜市", "老城小吃夜市", "車站商圈夜市", "廟口美食夜市"],
    "cafe_street": ["文青咖啡街", "老屋咖啡巷", "河岸咖啡街區", "文創甜點街", "巷弄咖啡聚落"],
    "historic_area": ["老街歷史街區", "古城文化街區", "傳統聚落保存區", "港町歷史街區", "紅磚建築街區"],
    "shopping_mall": ["大型購物中心", "車站百貨商圈", "室內娛樂商場", "複合式生活商場", "都會購物廣場"],
    "park": ["都會公園", "河濱公園", "森林公園", "親子休閒公園", "湖畔景觀公園"],
    "temple": ["百年古廟", "山城寺廟", "廟口文化廣場", "傳統信仰中心", "歷史寺院"],
}


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs or (DATA_DIR, FIGURE_DIR, MODEL_DIR):
        os.makedirs(directory, exist_ok=True)


def clip_score(x: Any) -> Any:
    return np.clip(x, 0, 5)


def clip_01(x: Any) -> Any:
    return np.clip(x, 0, 1)


def matrix_cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    numerator = left @ right.T
    denom = np.linalg.norm(left, axis=1, keepdims=True) * np.linalg.norm(right, axis=1)
    return np.divide(numerator, denom, out=np.zeros_like(numerator, dtype=float), where=denom != 0)


def vector_cosine_similarity(a: Any, b: Any) -> float:
    return float(matrix_cosine_similarity(np.asarray(a, dtype=float).reshape(1, -1), np.asarray(b, dtype=float).reshape(1, -1))[0, 0])


def softmax(scores: Any, temperature: float = 0.45) -> np.ndarray:
    values = np.asarray(scores, dtype=float) / temperature
    values = values - values.max()
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def fill_numeric_na(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = result.select_dtypes(include=np.number).columns
    result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())
    return result


def sample_if_needed(df: pd.DataFrame, max_rows: int | None, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(max_rows, random_state=random_state).reset_index(drop=True)


def primary_attraction_type(attraction_type: Any) -> str:
    if pd.isna(attraction_type):
        return "unknown"
    return str(attraction_type).split("+")[0]


def build_realistic_scene_name(attraction_id: int, attraction_type: str) -> str:
    primary_type = primary_attraction_type(attraction_type)
    pool = REALISTIC_SCENE_NAME_POOLS.get(primary_type, ["旅遊景點"])
    base_name = pool[int(attraction_id) % len(pool)]
    suffix = f"複合景點 {int(attraction_id):03d}" if "+" in str(attraction_type) else f"{int(attraction_id):03d}"
    return f"{base_name} {suffix}"


def add_scene_names(attractions: pd.DataFrame) -> pd.DataFrame:
    result = attractions.copy()
    if "realistic_scene_name" not in result.columns:
        result["realistic_scene_name"] = [
            build_realistic_scene_name(row.attraction_id, row.attraction_type)
            for row in result.itertuples(index=False)
        ]
    return result


def cluster_label(cluster_id: int) -> str:
    return CLUSTER_LABELS.get(int(cluster_id), f"Cluster {int(cluster_id)}")


def cluster_description(cluster_id: int) -> str:
    return CLUSTER_DESCRIPTIONS.get(int(cluster_id), "目前尚未定義此 cluster 的人工語意解釋。")


def cluster_title(cluster_id: int) -> str:
    return f"Cluster {int(cluster_id)}：{cluster_label(cluster_id)}"


def confidence_level(confidence: float) -> tuple[str, str]:
    confidence = float(confidence)
    if confidence >= 0.70:
        return "高信心", "模型對此使用者所屬 cluster 的判定較明確，可主要依據目前 cluster 解讀推薦結果。"
    if confidence >= 0.40:
        return "中等信心", "模型大致能判定使用者偏好，但仍建議參考 Soft Cluster Probability 中排名較高的其他 cluster。"
    return "低信心", "模型對單一 cluster 的判定不夠明確，表示使用者偏好可能分散於多個 cluster。建議不要只依賴目前 hard cluster 解釋。"


def load_required_csv(path: str | Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"找不到檔案：{path}")
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
