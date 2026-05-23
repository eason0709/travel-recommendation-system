# 2

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/travel_behavior.csv")

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
    "selected_popularity",
]

X = df[features]

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# =========================
# 用 BIC 自動選 GMM cluster 數
# =========================

candidate_ks = range(2, 9)

bic_scores = []

for k in candidate_ks:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42
    )
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))

best_k = candidate_ks[int(np.argmin(bic_scores))]

print()
print("BIC scores:")
for k, bic in zip(candidate_ks, bic_scores):
    print(f"k={k}, BIC={bic:.2f}")

print()
print("Best k:")
print(best_k)

gmm = GaussianMixture(
    n_components=best_k,
    covariance_type="full",
    random_state=42
)

gmm.fit(X_scaled)

soft_probs = gmm.predict_proba(X_scaled)
hard_labels = np.argmax(soft_probs, axis=1)
max_prob = soft_probs.max(axis=1)

df["soft_cluster"] = hard_labels
df["soft_cluster_confidence"] = max_prob

for i in range(best_k):
    df[f"cluster_prob_{i}"] = soft_probs[:, i]

# =========================
# 評估 soft cluster
# =========================

sil = silhouette_score(X_scaled, hard_labels)

print()
print("Silhouette Score:")
print(sil)

print()
print("Soft Cluster Distribution:")
print(df["soft_cluster"].value_counts().sort_index())

print()
print("Cluster Confidence Summary:")
print(df["soft_cluster_confidence"].describe())

# =========================
# Standardized cluster mean
# =========================

scaled_df = pd.DataFrame(X_scaled, columns=features)
scaled_df["soft_cluster"] = hard_labels

cluster_summary = scaled_df.groupby("soft_cluster")[features].mean()

cluster_summary.to_csv(
    "data/soft_cluster_feature_mean.csv",
    encoding="utf-8-sig"
)

# =========================
# 每群自動解釋
# =========================

interpret_rows = []

print()
print("Cluster Interpretation:")
print()

for cluster_id in cluster_summary.index:
    row = cluster_summary.loc[cluster_id]

    top_high = row.sort_values(ascending=False).head(5)
    top_low = row.sort_values(ascending=True).head(5)

    print(f"Cluster {cluster_id}")
    print("High features:")
    print(top_high)
    print()
    print("Low features:")
    print(top_low)
    print("-" * 60)

    for feature, value in top_high.items():
        interpret_rows.append({
            "cluster": cluster_id,
            "direction": "high",
            "feature": feature,
            "z_mean": value
        })

    for feature, value in top_low.items():
        interpret_rows.append({
            "cluster": cluster_id,
            "direction": "low",
            "feature": feature,
            "z_mean": value
        })

interpret_df = pd.DataFrame(interpret_rows)

interpret_df.to_csv(
    "data/soft_cluster_interpretation.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================
# Heatmap
# =========================

from matplotlib.colors import TwoSlopeNorm


def save_cluster_heatmap(cluster_summary, output_path):
    """
    Save an annotated red-blue heatmap for standardized cluster feature means.

    Red: feature mean is higher than the overall average.
    Blue: feature mean is lower than the overall average.
    Number in each cell: exact z_mean value rounded to 2 decimals.
    """

    values = cluster_summary.values.astype(float)

    # Use a symmetric range around zero so red/blue intensity is comparable.
    max_abs_value = max(
        2.0,
        float(np.nanmax(np.abs(values)))
    )

    norm = TwoSlopeNorm(
        vmin=-max_abs_value,
        vcenter=0.0,
        vmax=max_abs_value
    )

    fig_width = max(14, len(cluster_summary.columns) * 1.05)
    fig_height = max(6, len(cluster_summary.index) * 0.75)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        norm=norm
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Standardized Mean Value (z_mean)")

    ax.set_xticks(range(len(cluster_summary.columns)))
    ax.set_xticklabels(
        cluster_summary.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(range(len(cluster_summary.index)))
    ax.set_yticklabels(cluster_summary.index)

    ax.set_xlabel("Feature")
    ax.set_ylabel("Soft Cluster")
    ax.set_title("Soft Cluster Feature Mean Heatmap")

    # Draw cell boundaries for readability.
    ax.set_xticks(
        np.arange(-0.5, len(cluster_summary.columns), 1),
        minor=True
    )
    ax.set_yticks(
        np.arange(-0.5, len(cluster_summary.index), 1),
        minor=True
    )
    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.8
    )
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add exact z_mean values inside cells.
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            text_color = "white" if abs(value) >= max_abs_value * 0.45 else "black"

            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


save_cluster_heatmap(
    cluster_summary=cluster_summary,
    output_path="figures/soft_cluster_feature_heatmap.png"
)

# =========================
# Confidence distribution
# =========================

plt.figure(figsize=(8, 5))

plt.hist(
    df["soft_cluster_confidence"],
    bins=30
)

plt.title("Soft Cluster Confidence Distribution")
plt.xlabel("Max Cluster Probability")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "figures/soft_cluster_confidence_distribution.png",
    dpi=300
)

plt.show()

# =========================
# 儲存結果與模型
# =========================

df.to_csv(
    "data/travel_behavior_soft_clustered.csv",
    index=False,
    encoding="utf-8-sig"
)

joblib.dump(
    {
        "features": features,
        "imputer": imputer,
        "scaler": scaler,
        "gmm": gmm,
        "cluster_summary": cluster_summary
    },
    "models/soft_cluster_pipeline.pkl"
)

print()
print("Saved:")
print("data/travel_behavior_soft_clustered.csv")
print("data/soft_cluster_feature_mean.csv")
print("data/soft_cluster_interpretation.csv")
print("models/soft_cluster_pipeline.pkl")