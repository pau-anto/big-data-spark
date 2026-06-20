test webhook

# 🌷 Big Data Project - Tulips vs Lilies Classification

Image classification project (tulips/lilies) using a microservices architecture with PySpark, Scala ML, and Streamlit.

**Status:** 🚀 In Development

---

## 📊 Architecture

```
Input Image
    ↓
[µS 1] Parsing (Python/PySpark)
    ├─ Load image (PIL/OpenCV)
    ├─ Convert to numpy array
    └─ Save as NPZ
    ↓
[µS 2] Preprocessing (Python/PySpark)
    ├─ Load NPZ
    ├─ Convert to grayscale
    ├─ Normalize (0-1)
    └─ Save as NPZ
    ↓
[µS 3] ML Inference (Scala)
    ├─ Load model.pkl
    ├─ Run predictions
    └─ Output JSON (class + scores)
    ↓
[UI] Streamlit Visualization
    ├─ Display original image
    ├─ Display BW image
    ├─ Display predictions
    └─ Show confidence scores
```

---

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| µS 1 & 2 | Python + PySpark | Image loading & preprocessing |
| µS 3 | Scala + MLlib | ML model inference |
| UI | Streamlit | Web interface & visualization |
| Dataset | Images | ~100 MB (tulips/lilies) |
| Version Control | Git + GitHub | Collaboration & code management |

---

## 📁 Project Structure

```
big-data-tulips-lilies/
├── microservice_1/          # Image loading & parsing
│   ├── main.py
│   ├── requirements.txt
│   └── tests/
├── microservice_2/          # Image preprocessing
│   ├── main.py
│   ├── requirements.txt
│   └── tests/
├── microservice_3/          # ML inference (Scala)
│   ├── build.sbt
│   ├── src/main/scala/
│   └── tests/
├── streamlit_app/           # Web UI
│   ├── app.py
│   ├── requirements.txt
│   └── config.toml
├── models/                  # Trained ML models (not in git)
├── data/                    # Dataset (images, not in git)
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
├── README.md                # This file
├── .gitignore               # Git ignore rules
└── requirements-all.txt     # All dependencies
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Java 8+ (for Scala/Spark)
- Scala 2.12+
- Apache Spark 3.0+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/TONNOM/big-data-tulips-lilies.git
cd big-data-tulips-lilies
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements-all.txt
```

### 4. Download Dataset
```bash
bash scripts/download_dataset.sh
```

### 5. Run Microservices
Open 4 different terminals:

**Terminal 1 - Microservice 1:**
```bash
cd microservice_1
python main.py
```

**Terminal 2 - Microservice 2:**
```bash
cd microservice_2
python main.py
```

**Terminal 3 - Microservice 3:**
```bash
cd microservice_3
sbt run
```

**Terminal 4 - Streamlit UI:**
```bash
cd streamlit_app
streamlit run app.py
```

The UI should open at `http://localhost:8501`

---

## 👥 Team & Roles

| Role | Person | Responsibility | Tech Stack |
|------|--------|-----------------|-----------|
| **P1** | Infrastructure & Docs | Dataset, GitHub, documentation, deployment | Git, Markdown |
| **P2** | Microservices 1 & 2 | Image loading & preprocessing | Python, PySpark, OpenCV |
| **P3** | ML & Microservice 3 | Model training & inference | Python, Scala, TensorFlow/PyTorch |
| **P4** | Streamlit & Integration | Web UI, pipeline orchestration | Python, Streamlit |

---

## 📅 Timeline

| Phase | Duration | Focus |
|-------|----------|-------|
| **Phase 1** | W1-2 (May 26 - Jun 2) | Setup & environment |
| **Phase 2** | W3-5 (Jun 2 - Jun 30) | µS 1 & 2 (parsing/preprocessing) |
| **Phase 3** | W6-8 (Jun 23 - Jul 21) | ML training & µS 3 |
| **Phase 4** | W9-10 (Jul 14 - Jul 26) | Streamlit UI & final tests |

---

## 📝 Development Guidelines

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

## 🔄 Microservice Data Flow

### µS 1 → µS 2 Interface
```
Input: Images in data/images/
Output: NPZ files in output_parsed/
├─ filename: {original_name}.npz
├─ format: {image_array: numpy.ndarray}
└─ shape: (batch_size, height, width, channels)
```

### µS 2 → µS 3 Interface
```
Input: NPZ files from µS 2 output
Output: NPZ files in output_bw/
├─ filename: {original_name}_bw.npz
├─ format: {image_array: numpy.ndarray}
└─ shape: (batch_size, height, width) - grayscale
```

### µS 3 Output Format
```json
{
  "filename": "image_name.jpg",
  "prediction": "tulip",
  "confidence": 0.95,
  "scores": {
    "tulip": 0.95,
    "lily": 0.05
  }
}
```

---

*Academic Project — ESGI — Spark core — July 2026*
