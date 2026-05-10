# can be ignored

import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 建立資料夾
# =========================

os.makedirs("figures", exist_ok=True)

# =========================
# 讀取資料
# =========================

df = pd.read_csv("data/travel_behavior.csv")

print()
print("Dataset Shape:")
print(df.shape)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("First 5 Rows:")
print(df.head())

print()
print("Missing Values:")
print(df.isnull().sum())

# =========================
# Numerical Summary
# =========================

print()
print("Statistical Summary:")
print(df.describe())

# =========================
# Correlation Matrix
# =========================

corr = df.corr(numeric_only=True)

plt.figure(figsize=(16, 12))

im = plt.imshow(
    corr,
    cmap="coolwarm",
    aspect="auto",
    vmin=-1,
    vmax=1
)

plt.colorbar(im)

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

# 每格顯示數值
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):

        value = corr.iloc[i, j]

        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=7
        )

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "figures/correlation_matrix.png",
    dpi=300
)


# =========================
# Distribution Plots
# =========================

features = [
    "budget",
    "available_time",
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

for feature in features:

    plt.figure(figsize=(6, 4))

    plt.hist(
        df[feature].dropna(),
        bins=30
    )

    plt.title(f"{feature} Distribution")

    plt.xlabel(feature)
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        f"figures/{feature}_distribution.png",
        dpi=300
    )

    plt.close()

print()
print("Analyze completed.")