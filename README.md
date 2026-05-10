# Travel Recommendation System Ver1 create at 20260511 0233

本專案是一個以旅遊行為資料為基礎的推薦系統示範。系統會先產生模擬使用者資料與景點資料，接著使用 Gaussian Mixture Model 進行 soft clustering，最後根據使用者所屬 cluster 的平均行為特徵推薦適合的旅遊景點。

系統主要流程如下：

```text
generate_data
→ cluster_interpretation_soft
→ generate_attractions
→ generate_user_interactions
→ recommendation_engine
→ evaluate_clusters
→ app
```

`analyze_data.py` 為選配檔案，可用於前期資料觀察與視覺化，不是主流程必要步驟。

---

## Project Structure

```text
.
├── generate_data.py
├── cluster_interpretation_soft.py
├── generate_attractions.py
├── generate_user_interactions.py
├── recommendation_engine.py
├── evaluate_clusters.py
├── analyze_data.py
├── app.py
├── requirements.txt
├── data/
├── figures/
└── models/
```

說明：

| 檔案                             | 功能                                                     |
| -------------------------------- | -------------------------------------------------------- |
| `generate_data.py`               | 產生模擬使用者旅遊行為資料                               |
| `cluster_interpretation_soft.py` | 使用 GMM 進行 soft clustering，並輸出 cluster 解釋與模型 |
| `generate_attractions.py`        | 產生模擬景點資料                                         |
| `generate_user_interactions.py`  | 根據使用者與景點相似度產生模擬互動紀錄                   |
| `recommendation_engine.py`       | 產生推薦候選清單與 Top-K 推薦結果                        |
| `evaluate_clusters.py`           | 比較不同推薦方法的效果                                   |
| `analyze_data.py`                | 選配，用於資料分布與相關係數分析                         |
| `app.py`                         | Streamlit 網頁介面                                       |

---

## Installation 我還沒試過 先別自己亂試 lol

建議使用 Python 虛擬環境。

```bash
python -m venv venv
```

Windows：

```bash
venv\Scripts\activate
```

macOS / Linux：

```bash
source venv/bin/activate
```

安裝套件：

```bash
pip install -r requirements.txt
```

`requirements.txt` 目前包含：

```text
pandas
numpy
matplotlib
scikit-learn
streamlit
joblib
```

---

## How to Run

請依照以下順序執行。

---

## Main Pipeline

### Step A: Generate User Behavior Data

```bash
python generate_data.py
```

此步驟會產生模擬使用者旅遊行為資料。

輸出檔案：

```text
data/travel_behavior.csv
```

此資料包含使用者條件與行為特徵，例如：

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

此步驟會使用 Gaussian Mixture Model 對使用者進行 soft clustering。

主要工作包含：

- 使用 mean imputation 補缺失值
- 使用 StandardScaler 標準化資料
- 使用 BIC 自動選擇 GMM cluster 數量
- 產生每位使用者的 cluster probability
- 產生 hard label，也就是最大機率的 cluster
- 計算 cluster confidence
- 儲存 clustering model
- 輸出 cluster 解釋與圖表

輸出檔案：

```text
data/travel_behavior_soft_clustered.csv
data/soft_cluster_feature_mean.csv
data/soft_cluster_interpretation.csv
models/soft_cluster_pipeline.pkl
figures/soft_cluster_feature_heatmap.png
figures/soft_cluster_confidence_distribution.png
```

其中：

| 欄位                      | 說明                                |
| ------------------------- | ----------------------------------- |
| `soft_cluster`            | 使用者最可能所屬的 cluster          |
| `soft_cluster_confidence` | 該使用者被分到此 cluster 的信心分數 |
| `cluster_prob_i`          | 使用者屬於第 i 個 cluster 的機率    |

注意：如果你的檔名目前是 `cluster_interpretation_soft.py.py`，建議改成：

```text
cluster_interpretation_soft.py
```

否則執行時要使用實際檔名：

```bash
python cluster_interpretation_soft.py.py
```

---

### Step C: Generate Attraction Data

```bash
python generate_attractions.py
```

此步驟會產生模擬景點資料。

輸出檔案：

```text
data/attractions.csv
```

景點特徵包含：

```text
cost_level
indoor_score
photo_value
nature_value
culture_value
food_value
popularity
estimated_time
```

景點類型包含：

```text
museum
nature_trail
night_market
cafe_street
historic_area
shopping_mall
park
temple
```

程式中也會產生部分混合型景點，例如：

```text
cafe_street+historic_area
nature_trail+park
```

---

### Step D: Generate User Interactions

```bash
python generate_user_interactions.py
```

此步驟會根據使用者行為向量與景點向量的相似度，模擬使用者與景點之間的互動。

輸入檔案：

```text
data/travel_behavior_soft_clustered.csv
data/attractions.csv
```

輸出檔案：

```text
data/user_interactions.csv
```

輸出欄位包含：

| 欄位            | 說明                         |
| --------------- | ---------------------------- |
| `user_id`       | 使用者 ID                    |
| `soft_cluster`  | 使用者所屬 cluster           |
| `attraction_id` | 景點 ID                      |
| `rating`        | 模擬評分                     |
| `clicked`       | 是否點擊，rating >= 4 時為 1 |

注意：目前程式中 `DEBUG_MODE = True`，因此只會使用部分使用者與景點進行測試。若要使用完整資料，請將程式中的設定改成：

```python
DEBUG_MODE = False
```

---

### Step E: Recommendation Engine

```bash
python recommendation_engine.py
```

此步驟會產生推薦候選清單與每位使用者的 Top-K 推薦結果。

輸入檔案：

```text
data/travel_behavior_soft_clustered.csv
data/attractions.csv
data/user_interactions.csv
```

輸出檔案：

```text
data/all_recommendation_candidates.csv
data/recommendations.csv
```

系統會計算多種分數：

| 分數                   | 說明                                              |
| ---------------------- | ------------------------------------------------- |
| `personal_score`       | 使用者個人行為向量與景點向量的相似度              |
| `cluster_score`        | 使用者所屬 cluster 平均行為向量與景點向量的相似度 |
| `collaborative_score`  | 同 cluster 使用者對景點的平均 rating              |
| `hybrid_score`         | personal、cluster、collaborative 的混合分數       |
| `recommendation_score` | 最終推薦排序分數                                  |

注意：目前程式中 `DEBUG_MODE = True`。若要跑完整資料，請將：

```python
DEBUG_MODE = True
```

改成：

```python
DEBUG_MODE = False
```

---

### Step F: Evaluate Recommendation Methods

```bash
python evaluate_clusters.py
```

此步驟會比較不同推薦方法的效果。

輸入檔案：

```text
data/travel_behavior_soft_clustered.csv
data/attractions.csv
data/user_interactions.csv
data/all_recommendation_candidates.csv
```

輸出檔案：

```text
data/recommendation_method_comparison.csv
figures/recommendation_method_comparison.png
```

比較方法包含：

| 方法                 | 說明                                       |
| -------------------- | ------------------------------------------ |
| `personal_only`      | 只使用單一使用者行為向量推薦               |
| `cluster_only`       | 只使用 cluster 平均行為向量推薦            |
| `collaborative_only` | 只使用同 cluster 使用者的歷史互動推薦      |
| `hybrid`             | 結合 personal、cluster、collaborative 推薦 |

評估指標包含：

| 指標                  | 說明                                       |
| --------------------- | ------------------------------------------ |
| `mean_similarity`     | Top-K 推薦景點與使用者行為向量的平均相似度 |
| `random_baseline`     | 隨機推薦的平均相似度                       |
| `recommendation_lift` | 推薦方法相對 random baseline 的提升幅度    |

---

### Step G: Run Streamlit App

```bash
streamlit run app.py
```

啟動後會開啟旅遊推薦系統網頁介面。

App 主要功能包含：

- 手動輸入使用者條件
- 使用既有使用者資料
- 預測使用者 soft cluster
- 顯示 cluster confidence
- 顯示 cluster probability
- 根據 cluster 平均行為特徵推薦景點
- 顯示推薦結果解釋
- 顯示 cluster 解釋
- 顯示推薦方法比較
- 顯示分析圖表

目前 App 中的推薦邏輯主要採用 cluster-based 推薦，也就是使用者被分到某個 cluster 後，系統會使用該 cluster 的平均行為特徵與景點特徵做相似度比較。

---

## Optional: Analyze Data

```bash
python analyze_data.py
```

此步驟不是主流程必要步驟。

它主要用來觀察資料分布與特徵相關性。

輸出內容包含：

```text
figures/correlation_matrix.png
figures/{feature}_distribution.png
```

用途包含：

- 檢查資料欄位
- 檢查 missing value
- 查看 numerical summary
- 畫出 correlation matrix
- 畫出各特徵分布圖

---

## Recommended Execution Order

建議第一次執行時使用以下完整順序：

```bash
python generate_data.py
python cluster_interpretation_soft.py
python generate_attractions.py
python generate_user_interactions.py
python recommendation_engine.py
python evaluate_clusters.py
streamlit run app.py
```

如果你的 clustering 檔案仍然叫做 `cluster_interpretation_soft.py.py`，請改用：

```bash
python generate_data.py
python cluster_interpretation_soft.py.py
python generate_attractions.py
python generate_user_interactions.py
python recommendation_engine.py
python evaluate_clusters.py
streamlit run app.py
```

如果要加上選配分析：

```bash
python generate_data.py
python analyze_data.py
python cluster_interpretation_soft.py
python generate_attractions.py
python generate_user_interactions.py
python recommendation_engine.py
python evaluate_clusters.py
streamlit run app.py
```

---

## Data Flow

```text
generate_data.py
    ↓
data/travel_behavior.csv
    ↓
cluster_interpretation_soft.py
    ↓
data/travel_behavior_soft_clustered.csv
data/soft_cluster_interpretation.csv
models/soft_cluster_pipeline.pkl
    ↓
generate_attractions.py
    ↓
data/attractions.csv
    ↓
generate_user_interactions.py
    ↓
data/user_interactions.csv
    ↓
recommendation_engine.py
    ↓
data/all_recommendation_candidates.csv
data/recommendations.csv
    ↓
evaluate_clusters.py
    ↓
data/recommendation_method_comparison.csv
figures/recommendation_method_comparison.png
    ↓
app.py
```

---

## Notes

本專案目前使用模擬資料，不是真實旅遊平台資料。

目前推薦系統的 App 版本偏向展示 cluster-based recommendation，也就是使用 cluster 的平均行為特徵來推薦景點，而不是直接用單一使用者行為向量排序。

如果要讓 App 使用 hybrid recommendation，需要進一步修改 `app.py`，讓它讀取 `recommendation_engine.py` 產生的 `hybrid_score` 或重新在 App 內計算 hybrid score。

部分程式目前設定為 `DEBUG_MODE = True`，因此只會處理部分資料。若要執行完整資料，請確認下列檔案中的 `DEBUG_MODE` 設定：

```text
generate_user_interactions.py
recommendation_engine.py
evaluate_clusters.py
```

將其改為：

```python
DEBUG_MODE = False
```

---

## Main Idea

本系統的核心概念是：

```text
使用者行為資料
→ soft clustering
→ 取得使用者所屬 cluster
→ 使用 cluster 平均偏好建立推薦基準
→ 將景點特徵與 cluster 特徵比較
→ 輸出相似度最高的旅遊景點
```

這種設計適合用來展示非監督式學習在推薦系統中的應用，尤其是當沒有明確標籤，但有使用者行為特徵時，可以先透過 clustering 找出相似族群，再根據族群特徵做推薦。
