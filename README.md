# 🔬 Sickle Cell Disease Detection & Clinical Risk Assessment System


> **An explainable, multi-stage clinical AI platform that combines Deep Learning, Explainable AI (Grad-CAM), and Computer Vision Morphometry to detect Sickle Cell Disease (SCD) and assess acute vaso-occlusion crisis risk.**

🌐 **Live Demo:** [sickle-cell-disease-detector-application.onrender.com](https://sickle-cell-disease-detector-application.onrender.com)

## 🩺 Project Overview

Sickle Cell Disease (SCD) is a widespread genetic blood disorder characterized by stiff, crescent-shaped red blood cells (RBCs). These rigid cells can cluster and obstruct microvasculature, leading to acute **vaso-occlusive pain crises (VOC)** and organ ischemia.

### The Problem
Traditional manual microscopy is labor-intensive, operator-dependent, and qualitatively subjective. Standard "black-box" deep learning models, while accurate at classification, do not explain their reasoning, cannot count individual abnormal cells, and fail to assess a patient's immediate physiological risk.

### The Solution
This platform bridges the gap between deep learning and clinical decision-making:
1. **Detects** disease presence using a fine-tuned MobileNetV1 CNN.
2. **Explains** its reasoning via Grad-CAM spatial visual attention heatmaps.
3. **Quantifies** individual cell morphology, counts, and spatial clustering using OpenCV Watershed morphometry.
4. **Calculates** a multi-factor clinical risk score (VORS) combining microscopic findings with patient bedside vitals ($\text{SpO}_2$, Age, Pain Crises history).

---

## ✨ Key Features

- **⚡ Lightweight & Fast:** Powered by MobileNetV1 with depthwise separable convolutions, optimized to run under 3 seconds per smear.
- **🔍 Explainable AI (XAI):** Grad-CAM overlay guarantees transparent validation by highlighting the exact microscopic regions that drove the model's prediction.
- **🔬 Precise RBC Morphometry:** Automated separation of overlapping cells via Watershed segmentation, with aspect-ratio and circularity classification.
- **🩸 Obstruction Risk Clustering:** Flags sickle cell clusters capable of capillary occlusion.
- **📊 Vaso-Occlusion Risk Engine (VORS):** Heuristic risk modeling that stratifies patient risk into *Low*, *Moderate*, and *High* crisis categories.
- **📄 One-Click PDF Export:** Generates clean, medical-grade diagnostic PDF reports for clinical records.

---

## 🏗 System Architecture (The 3-Gate Pipeline)

```
       [ Microscopic Smear Image ]          [ Patient Vitals: Age, SpO2, Pain Crises ]
                    │                                             │
                    ▼                                             │
      ┌──────────────────────────┐                                │
      │ Gate 1: MobileNetV1 CNN  │                                │
      │  & Grad-CAM Heatmap      │                                │
      └─────────────┬────────────┘                                │
                    ▼                                             │
      ┌──────────────────────────┐                                │
      │ Gate 2: OpenCV Morphology│                                │
      │ (Watershed, SCI %, CD %) │                                │
      └─────────────┬────────────┘                                │
                    ▼                                             │
      ┌───────────────────────────────────────────────────────────┴───┐
      │ Gate 3: Clinical Risk Engine (IBSI, Hb Range, VORS Risk)      │
      └─────────────────────────────┬─────────────────────────────────┘
                                    ▼
       [ Interactive Clinical Web Dashboard & Exportable PDF Report ]
```

### 1. Gate 1 — Deep Learning & Attention (MobileNetV1 + Grad-CAM)
- The input smear is resized to $224 \times 224$ and passed through MobileNetV1.
- Yields a binary classification (`Positive` vs. `Negative`) with confidence metrics.
- Grad-CAM extracts gradients from the final convolutional layer, projecting a heatmap of attention over the original image.

### 2. Gate 2 — Microscopic Cell Morphometry (OpenCV)
- **Segmentation:** Green-channel extraction, Otsu thresholding, morphological opening, and distance-transform Watershed splitting.
- **Debris Filtering:** Excludes platelets, WBCs, clumps, and border-cut cells.
- **Sickle Detection Rule:**
  $$\text{Aspect Ratio} \ge 2.0 \quad \text{and} \quad \text{Circularity} \le 0.65$$
- **Clustering:** Sickle cells within $1.1 \times$ the average RBC diameter are flagged as clusters (magenta overlay).

### 3. Gate 3 — Clinical Risk Engine (VORS & Severity)
- Integrates patient demographics and vitals with image-derived indices to evaluate risk.

---

## 📐 Mathematical & Clinical Formulations

### Sickle Cell Index (SCI)
Measures the proportion of sickled cells in the field of view:
$$\text{SCI} = \left( \frac{\text{Total Sickled RBCs}}{\text{Total Segmented RBCs}} \right) \times 100$$

### Clustering Density (CD)
Quantifies the fraction of sickle cells actively forming aggregations:
$$\text{CD} = \left( \frac{\text{Clustered Sickle Cells}}{\text{Total Sickle Cells}} \right) \times 100$$

### Image-Based Severity Index (IBSI)
Combines abundance and clumping:
$$\text{IBSI} = 0.70 \times \text{SCI} + 0.30 \times \text{CD}$$
* **Mild:** $\text{IBSI} < 34$
* **Moderate:** $34 \le \text{IBSI} < 67$
* **Severe:** $\text{IBSI} \ge 67$

### Vaso-Occlusion Risk Score (VORS)
Calculates emergency vaso-occlusion crisis likelihood:
$$\text{VORS} = 0.40 \times \text{Score}_{\text{SpO}_2} + 0.30 \times \text{IBSI} + 0.20 \times \text{Score}_{\text{Hb}} + 0.10 \times \text{Score}_{\text{Pain}}$$
* **Low Risk:** $< 34$
* **Moderate Risk:** $34 \text{ to } < 67$
* **High Risk:** $\ge 67$

---

## 💻 Clinical Dashboard & Web UI

The web application features:
* **Interactive Smear Upload:** Drag-and-drop support with real-time preview.
* **Synchronized Dual Visualizations:**
  * Left: Grad-CAM attention heatmap (AI perspective).
  * Right: OpenCV segmentation overlay with color-coded boundaries (Green: Normal RBCs, Red: Sickle cells, Magenta: Clusters).
* **Clinical Vitals Input:** Real-time adaptation based on patient age, biological sex, $\text{SpO}_2$, and historical pain crises.
* **Instant PDF Download:** Embedded client-side PDF generator (`html2pdf.js`).

---

## 📂 Repository Structure

```text
├── app/
│   ├── backend/
│   │   ├── app.py             # Flask web server & REST API
│   │   └── inference.py       # Unified multi-stage inference pipeline
│   └── frontend/
│       ├── static/
│       │   ├── css/style.css  # Responsive medical dashboard design
│       │   └── js/app.js      # Fetch API, DOM updates, PDF generation
│       └── templates/
│           └── index.html     # Clinical dashboard layout
├── src/
│   ├── cell_analysis.py       # OpenCV Watershed segmentation & clustering
│   ├── gradcam.py             # Grad-CAM heatmap backpropagation
│   └── severity.py            # IBSI, Hb estimation, and VORS calculations
├── models/
│   └── paper_style_mobilenetv1_best.keras # Fine-tuned MobileNetV1 weights
├── config.py                  # Centralized clinical thresholds & weights
├── Dockerfile                 # Production Docker container definition
├── requirements.txt           # Python application dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Local Setup & Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git & Git LFS

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/sickle-cell-detector.git
cd sickle-cell-detector
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Server
```bash
python app/backend/app.py
```
Open your browser and navigate to **`http://localhost:7860`**.

---

## 🐳 Cloud Deployment (Docker & Render)

This repository includes a production-ready `Dockerfile` optimized for memory-constrained cloud environments (such as Render's Free Tier):

```bash
# Build the Docker image
docker build -t sickle-cell-detector .

# Run the container
docker run -p 7860:7860 sickle-cell-detector
```

When deploying on **Render.com**:
1. Connect this repository as a **Web Service**.
2. Select **Runtime: Docker**.
3. Select **Instance Type: Free**.
4. Render builds and launches the application automatically!

---

## 🛠 Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Deep Learning** | TensorFlow / Keras, MobileNetV1, Grad-CAM (XAI) |
| **Computer Vision** | OpenCV, SciPy (Distance Transform, Connected Components) |
| **Backend & API** | Python 3.11, Flask, Gunicorn |
| **Frontend** | Semantic HTML5, Modern CSS3 (Grid & Flexbox), Vanilla JavaScript (ES6+) |
| **Reporting** | html2pdf.js |
| **Deployment** | Docker/ Render Cloud Platform |
