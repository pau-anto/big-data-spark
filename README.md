# 🌷 Big Data Project - Tulips vs Lilies Classification

Image classification project (tulips/lilies) using a single end-to-end PySpark pipeline, with ML inference and a Streamlit dashboard.

**Status:** 🚀 In Development — parsing step implemented, preprocessing/training/inference/visualization still to come.

---

## 📊 Architecture

```
                  ┌──────────────────────────────────────────┐
                  │              pipeline.py                  │
                  │           (single PySpark script)          │
                  │                                              │
   data/Train/   ─┤  [1] Parsing            ✅ implemented        │
   data/Test/     │      ├─ spark.read.format("binaryFile")        │
                  │      ├─ decode_image_bytes (UDF, per worker)     │
                  │      ├─ resize to 64×64, extract RGB pixels        │
                  │      └─ train_parsed_df / test_parsed_df (Spark DF)│
                  │                ↓                                  │
                  │  [2] Preprocessing       ⏳ to do                   │
                  │      ├─ grayscale + normalize (0-1)                  │
                  │                ↓                                      │
                  │  [3] Training            ⏳ to do                       │
                  │      └─ train ML model on train_parsed_df                │
                  │                ↓                                          │
                  │  [4] Inference           ⏳ to do                           │
                  │      └─ predict on test_parsed_df → output/predictions/      │
                  └──────────────────────────────────────────────────────────────┘
                                ↓
                  ┌──────────────────────────────────────────┐
                  │         streamlit_app/app.py    ⏳ to do    │
                  │  Currently: simple image uploader/viewer     │
                  │  Target: display predictions + confidence       │
                  └──────────────────────────────────────────────┘
```

**Principle:** maximize what runs through Spark (parsing, preprocessing, batch I/O) and rely on a Spark DataFrame throughout — no `collect()` / `toPandas()` / `toList()` except for final display. Only the ML training/inference step is expected to fall back to plain Python, since Spark/MLlib isn't well suited to an arbitrary pickled model.

---

## 🛠️ Technical Stack

| Component | Technology | Status |
|-----------|-----------|--------|
| Parsing | PySpark (`binaryFile` + UDF) | ✅ Done |
| Preprocessing (grayscale, normalize) | PySpark | ⏳ To do |
| Training | Python (likely scikit-learn) | ⏳ To do |
| Inference | Python | ⏳ To do |
| UI | Streamlit | ⏳ Basic uploader only |
| Dataset | Images (tulips/lilies) | ✅ In `data/` |
| Version Control | Git + GitHub | ✅ |

---

## 📁 Project Structure (actual, as in the repo)

```
big-data-spark/
├── .vscode/
│   └── settings.json
├── data/                       # Dataset (images)
│   ├── Train/
│   │   ├── lys/
│   │   └── tulipes/
│   ├── Test/
│   │   ├── lys/
│   │   └── tulipes/
│   ├── Train_5/                # Small subsets for quick local testing
│   │   ├── lys/
│   │   └── tulipes/
│   └── Test_5/
│       ├── lys/
│       └── tulipes/
├── streamlit_app/
│   └── app.py                  # Currently: image upload + display only
├── notebook.ipynb               # Source notebook (pipeline.py is exported from this)
├── pipeline.py                  # Single end-to-end PySpark script (.py export of the notebook)
├── .gitignore
└── README.md                     # This file
```

> `output/` (predictions, trained model) is referenced in `pipeline.py` but not yet created/committed — it will hold the pipeline's results once training/inference are implemented.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Java 8+ (required by Spark)
- PySpark installed (`pip install pyspark`)
- Pillow (`pip install pillow`)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/pau-anto/big-data-spark.git
cd big-data-spark
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pyspark pillow streamlit
```

### 3. Run the Pipeline (parsing step only, for now)
```bash
python pipeline.py
```
This reads images from `data/Train/` and `data/Test/`, decodes/resizes them (64×64 RGB) in a distributed way, and prints the row counts of `train_parsed_df` and `test_parsed_df`. Nothing is written to disk yet — preprocessing, training, and inference are still to be added.

> Tip: for quick local iteration, point `TRAIN_PATH` / `TEST_PATH` at `data/Train_5/` and `data/Test_5/` (small subsets) instead of the full dataset.

### 4. Run the Streamlit UI (placeholder for now)
```bash
cd streamlit_app
streamlit run app.py
```
The UI should open at `http://localhost:8501`. Currently only supports uploading and previewing an image — not yet wired to the pipeline's predictions.

---

## 🔄 Pipeline Steps (in `pipeline.py`)

### 0 · Spark session & imports
Creates the `SparkSession`, configures driver/executor memory, sets log level to `WARN`.

### 1 · Paths & constants
```python
TRAIN_PATH   = "data/Train/"
TEST_PATH    = "data/Test/"
OUTPUT_PREDS = "output/predictions/"
MODEL_PATH   = "output/model/"
TARGET_SIZE  = (64, 64)
```

### 2 · Parsing ✅
- Reads images with `spark.read.format("binaryFile")` (recursive lookup, filtered by extension).
- `decode_image_bytes` is registered as a Spark UDF: runs on each worker, resizes the image to 64×64 with PIL, and returns flattened RGB pixel values — nothing is pulled back to the driver.
- Extracts `image_id` and `label` (`lys` / `tulipes`) from the file path.
- Produces `train_parsed_df` and `test_parsed_df`, two Spark DataFrames with columns `image_id`, `label`, `pixels`.

### 3 · Preprocessing — ⏳ not implemented yet
Planned: convert to grayscale and normalize pixel values to `[0.0, 1.0]`.

### 4 · Training — ⏳ not implemented yet
Planned: train a classifier (tulipe vs lys) on `train_parsed_df`, save it to `output/model/`.

### 5 · Inference — ⏳ not implemented yet
Planned: run the trained model on `test_parsed_df`, write predictions (class + confidence) to `output/predictions/` as Parquet/JSON.

### 6 · Visualization — ⏳ not implemented yet
Planned: Streamlit app reads `output/predictions/` and displays images alongside predicted class and confidence.

---

## 📝 Development Guidelines

### ⚠️ `collect()` is forbidden

`collect()` is banned everywhere in this codebase, and this is not a style preference — it's a correctness rule.

- **`count()`** ships one integer back to the driver (the edge node). Cheap, no matter the dataset size.
- **`collect()`** pulls the *entire* distributed dataset off the cluster's executors, converts it from Spark's RDD/DataFrame representation into a plain Python list, and materializes it on the driver machine.

Why this matters: if the pipeline is processing 10 TB of images, calling `collect()` tries to load all 10 TB into the memory of a single machine (the driver / edge node). That machine runs out of RAM and crashes — and in the worst case it can also destabilize the cluster it just pulled all that data from, since the collect happens on the edge, dragging data off the cluster and onto the edge, making both unusable. `count()` never has this problem because it never leaves the cluster with more than a number.

**Rule applied in `pipeline.py`:**
- No `collect()` / `toPandas()` / `toList()` anywhere in the transformation logic.
- The only place plain Python (and small local objects) is allowed is the very last step — final display / visualization in Streamlit — never in parsing, preprocessing, or batch I/O.
- If a step "needs" the data locally, that's a signal the step should be redesigned to stay in Spark, not an excuse to call `collect()`.

This is graded as a hard requirement: **a fully Spark-based pipeline is the deliverable — a version that quietly falls back to full Python everywhere scores 0**, even if it produces the right output.

### Git Workflow
```bash
# Create your feature branch
git checkout -b feat/your-feature

# Make changes
# ... edit files ...

# Commit with clear messages
git commit -m "feat: describe what you added"

# Push to GitHub
git push origin feat/your-feature

# Create Pull Request for review
```

---

*Academic Project — ESGI — Spark core — July 2026*
