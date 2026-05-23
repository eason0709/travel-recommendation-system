# Travel Recommendation System

本專案是一個以模擬旅遊行為資料為基礎的推薦系統示範。系統會產生使用者旅遊行為資料與景點資料，使用 Gaussian Mixture Model (GMM) 進行 soft clustering，並在 Streamlit App 中展示不同推薦模式、cluster 解釋、模型信心判讀與推薦結果。

本專案的重點不是使用真實旅遊平台資料，而是展示非監督式分群如何輔助推薦系統的設計、解釋與決策流程。

---

## 專案重點

| 功能 | 說明 |
|---|---|
| 使用者資料生成 | 產生模擬旅遊偏好、預算、時間、天氣、疲勞程度等資料 |
| 景點資料生成 | 產生模擬景點特徵，並提供接近現實場景的顯示名稱 |
| GMM Soft Clustering | 使用 Gaussian Mixture Model 將使用者分成多個 soft cluster |
| Cluster Probability | 顯示使用者屬於每個 cluster 的機率 |
| Cluster Confidence | 判斷模型對單一 cluster 的判定是否明確 |
| Low Confidence Warning | 當模型信心不足時，提醒使用者不要只依賴單一 hard cluster |
| Recommendation Mode | App 可選擇 `personal_only` 或 `cluster_only` 推薦模式 |
| 推薦結果解釋 | 顯示推薦景點與使用者或 cluster 偏好的關係 |
| 方法比較 | 比較 personal-only、cluster-only、collaborative-only、hybrid 的推薦效果 |
| 視覺化 | 產生 cluster heatmap、confidence distribution、推薦方法比較圖 |

---

## Project Structure

```text
travel_clustering_project/
├── travel_utils.py
├── generate_data.py
├── cluster_interpretation_soft.py
├── generate_attractions.py
├── generate_user_interactions.py
├── recommendation_engine.py
├── evaluate_clusters.py
├── analyze_data.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/       # 執行後產生，建議不要上傳 GitHub
├── figures/    # 執行後產生，建議不要上傳 GitHub
└── models/     # 執行後產生，建議不要上傳 GitHub
```

---

## 檔案說明

| 檔案 | 功能 |
|---|---|
| `travel_utils.py` | 共用工具檔，集中管理特徵欄位、cluster 標籤、場景名稱、相似度計算與 confidence 判讀 |
| `generate_data.py` | 產生模擬使用者旅遊行為資料 |
| `cluster_interpretation_soft.py` | 使用 GMM 進行 soft clustering，輸出 cluster probability、confidence、模型與 heatmap |
| `generate_attractions.py` | 產生模擬景點資料與可讀的場景名稱 |
| `generate_user_interactions.py` | 根據使用者與景點相似度產生模擬互動紀錄 |
| `recommendation_engine.py` | 產生推薦候選清單與 Top-K 推薦結果 |
| `evaluate_clusters.py` | 比較不同推薦方法的效果 |
| `analyze_data.py` | 選配，用於資料分布與相關係數分析 |
| `app.py` | Streamlit 網頁展示系統 |

---

## Installation

建議使用 Python 虛擬環境。

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

安裝套件:

```bash
pip install -r requirements.txt
```

`requirements.txt` 建議包含:

```text
pandas
numpy
matplotlib
scikit-learn
streamlit
joblib
pyarrow
```

其中 `pyarrow` 是為了避免讀取部分 pandas / joblib 物件時出現缺少套件的問題。

---

## 執行順序

第一次執行建議依照以下流程:

```bash
python generate_data.py
python cluster_interpretation_soft.py
python generate_attractions.py
python generate_user_interactions.py
python recommendation_engine.py
python evaluate_clusters.py
streamlit run app.py
```

如果只想啟動 App，前提是 `data/`、`models/`、`figures/` 已經存在:

```bash
streamlit run app.py
```

---

## Main Pipeline

### Step A: Generate User Behavior Data

```bash
python generate_data.py
```

輸出:

```text
data/travel_behavior.csv
```

主要欄位包含:

```text
budget
available_time
weather_badness
social_context
fatigue
spontaneity
selected_indoor_score
selected_cost_level
selected_photo_value
selected_nature_value
selected_culture_value
selected_food_value
selected_popularity
```

---

### Step B: Soft Clustering and Cluster Interpretation

```bash
python cluster_interpretation_soft.py
```

此步驟會使用 GMM 進行 soft clustering，並輸出每位使用者屬於各 cluster 的機率。

輸出:

```text
data/travel_behavior_soft_clustered.csv
data/soft_cluster_feature_mean.csv
data/soft_cluster_interpretation.csv
data/soft_cluster_label_summary.csv
models/soft_cluster_pipeline.pkl
figures/soft_cluster_feature_heatmap.png
figures/soft_cluster_confidence_distribution.png
```

重要欄位:

| 欄位 | 說明 |
|---|---|
| `soft_cluster` | 使用者最可能所屬的 cluster |
| `soft_cluster_confidence` | 最高 cluster probability，也就是模型信心 |
| `cluster_prob_i` | 使用者屬於第 i 個 cluster 的機率 |

`soft_cluster_feature_heatmap.png` 使用紅藍漸層呈現標準化後的 cluster feature mean。紅色代表高於整體平均，藍色代表低於整體平均，格子內數字為 `z_mean`。

---

### Step C: Generate Attraction Data

```bash
python generate_attractions.py
```

輸出:

```text
data/attractions.csv
```

景點欄位包含:

```text
attraction_id
name
realistic_scene_name
attraction_type
cost_level
indoor_score
photo_value
nature_value
culture_value
food_value
popularity
estimated_time
```

`realistic_scene_name` 是展示用的現實場景式名稱，例如老街、咖啡街、博物館、夜市等。這些名稱不是從真實旅遊平台爬取而來，仍屬於模擬資料。

---

### Step D: Generate User Interactions

```bash
python generate_user_interactions.py
```

輸出:

```text
data/user_interactions.csv
```

此步驟會根據使用者行為向量與景點特徵相似度，產生模擬 rating 與 clicked 紀錄。

---

### Step E: Recommendation Engine

```bash
python recommendation_engine.py
```

輸出:

```text
data/all_recommendation_candidates.csv
data/recommendations.csv
```

分數包含:

| 分數 | 說明 |
|---|---|
| `personal_score` | 使用者個人行為向量與景點向量的相似度 |
| `cluster_score` | 使用者所屬 cluster 平均行為向量與景點向量的相似度 |
| `collaborative_score` | 同 cluster 使用者對景點的平均 rating |
| `hybrid_score` | personal、cluster、collaborative 的混合分數 |

---

### Step F: Evaluate Recommendation Methods

```bash
python evaluate_clusters.py
```

輸出:

```text
data/recommendation_method_comparison.csv
figures/recommendation_method_comparison.png
```

比較方法:

| 方法 | 說明 |
|---|---|
| `personal_only` | 只使用個人行為向量推薦 |
| `cluster_only` | 只使用 cluster 平均行為向量推薦 |
| `collaborative_only` | 只使用同 cluster 使用者歷史互動推薦 |
| `hybrid` | 結合 personal、cluster、collaborative 分數 |

目前完整資料測試結果顯示，`personal_only` 的表現最好，`hybrid` 接近但沒有超過。這表示在目前模擬資料與評估方式下，個人行為向量是最直接有效的推薦依據；cluster-based 方法則較適合作為可解釋推薦與冷啟動輔助。

---

### Step G: Run Streamlit App

```bash
streamlit run app.py
```

App 功能包含:

| 功能 | 說明 |
|---|---|
| 手動輸入使用者條件 | 使用 sidebar 輸入預算、時間、天氣、偏好等 |
| 使用既有使用者 | 從已產生資料中選擇 user ID |
| Recommendation Mode | 可選 `personal_only` 或 `cluster_only` |
| Soft Cluster | 顯示模型判定的 cluster |
| Cluster Confidence | 顯示模型判定信心 |
| Soft Cluster Probability | 顯示使用者屬於各 cluster 的機率 |
| Low Confidence Warning | 當模型信心較低時顯示提醒 |
| Cluster 計算流程說明 | 解釋 imputer、scaler、GMM probability 與 confidence 的計算流程 |
| 推薦景點 | 顯示推薦景點與現實場景式名稱 |
| 推薦結果解釋 | 說明推薦分數來源 |
| Cluster 解釋 | 顯示該 cluster 高於或低於平均的特徵 |
| 分析圖表 | 顯示 heatmap、confidence distribution、method comparison 等 |

---

## Recommendation Mode 說明

| 模式 | 計算方式 | 適合用途 |
|---|---|---|
| `personal_only` | 使用個人行為向量與景點特徵直接計算相似度 | 目前評估結果中表現最好，較適合個人化推薦 |
| `cluster_only` | 使用所屬 cluster 的平均行為特徵與景點特徵計算相似度 | 較適合展示 cluster 解釋、族群偏好與冷啟動概念 |

---

## Cluster 計算流程

本系統的 cluster 不是人工 if-else 規則，而是由 GMM 根據使用者特徵計算機率。

流程如下:

```text
使用者輸入條件
→ 補缺失值 imputation
→ 標準化 StandardScaler
→ 輸入 Gaussian Mixture Model
→ 計算各 cluster probability
→ 取 probability 最大者作為 soft_cluster
→ 最大 probability 作為 cluster confidence
```

當 `cluster confidence` 較低時，表示模型對單一 cluster 的判定不夠明確，使用者可能同時具有多個 cluster 的特徵。因此 App 會提醒使用者參考 Soft Cluster Probability，而不是只依賴單一 hard cluster。

---

## Cluster Labels

本專案不再在 README 內固定寫死 `cluster 0~7` 的中文名稱。原因是 GMM 每次重新訓練或資料生成邏輯調整後，cluster 編號可能重新排列。例如某次結果中 `cluster 0` 的 `budget` 標準化平均值可能偏高，此時若仍把它寫成「低預算型」就會和 heatmap 衝突。

目前 cluster label 由 `cluster_interpretation_soft.py` 根據當次產生的 `data/soft_cluster_feature_mean.csv` 自動推論，並輸出到:

```text
data/soft_cluster_label_summary.csv
```

該檔案包含:

| 欄位 | 說明 |
|---|---|
| `cluster` | cluster 編號 |
| `cluster_label` | 根據目前標準化特徵平均值推論出的中文名稱 |
| `description` | 根據高於平均與低於平均特徵產生的說明 |
| `top_high_features` | 該 cluster 最高的幾個 z_mean 特徵 |
| `top_low_features` | 該 cluster 最低的幾個 z_mean 特徵 |

App 會優先讀取 `data/soft_cluster_label_summary.csv` 來顯示 cluster 名稱與說明。因此，若重新執行 clustering，請重新執行:

```bash
python cluster_interpretation_soft.py
```

這樣 App 與 heatmap 會使用同一份 clustering 結果，避免 cluster label 和 heatmap 數值不一致。

---

## Optional: Analyze Data

```bash
python analyze_data.py
```

此步驟不是主流程必要步驟，用於產生資料分布與相關係數圖。

輸出:

```text
figures/correlation_matrix.png
figures/{feature}_distribution.png
```

---

## GitHub 注意事項

`data/`、`figures/`、`models/` 屬於執行後產生的資料、圖片與模型檔。若不想把大量產物或模型檔放到 GitHub，建議在 `.gitignore` 中排除:

```gitignore
data/
figures/
models/
__pycache__/
*.pyc
venv/
.venv/
.env
.DS_Store
Thumbs.db
```

如果這些資料夾已經被 Git 追蹤，需要使用以下指令從 Git 追蹤中移除，但保留本機檔案:

```bash
git rm -r --cached data figures models
```

之後再 commit 並 push。

---

## Notes

本專案目前使用模擬資料，不是真實旅遊平台資料。因此結果主要用於展示推薦系統流程、soft clustering 解釋、模型信心判讀與推薦模式比較，不應直接解讀為真實旅遊平台上的實際推薦成效。
