# 🚗 Predictive Vehicle Maintenance & Explainable Diagnostics Platform

A production-grade AI system for predicting vehicle maintenance needs with full SHAP explainability, health-based RUL estimation, rule-based diagnostics, and a conversational AI assistant.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd vehicle_maintenance_platform
pip install -r requirements.txt
```

### 2. Configure API Key (Optional — for AI Chat)

Edit the `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a **free** Groq API key at: https://console.groq.com/

> The platform works fully without an API key — only the AI chat assistant requires one.

### 3. Launch the App

```bash
python app.py
```

Then open: **http://localhost:7860**

> **First launch:** The system will automatically train the Random Forest model on the dataset (~1-2 minutes for 50K records). Subsequent launches load from cache instantly.

---

## 📁 Project Structure

```
vehicle_maintenance_platform/
│
├── app.py                  ← Main Gradio dashboard (entry point)
├── preprocessing.py        ← Data preprocessing pipeline
├── train_model.py          ← Random Forest training & evaluation
├── explainability.py       ← SHAP TreeExplainer module
├── diagnostic_engine.py    ← Rule-based subsystem diagnostics
├── rul_engine.py           ← Health-based RUL estimation
├── ai_agent.py             ← Groq/OpenAI conversational assistant
├── visualizations.py       ← Plotly chart generators
├── cache_manager.py        ← Global model cache
├── utils.py                ← Shared utilities & HTML builders
│
├── models/                 ← Saved model artifacts (auto-created)
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── shap_explainer.pkl
│   └── metrics.json
│
├── data/
│   └── vehicle_maintenance_data.csv
│
├── assets/                 ← Logs and generated assets
├── requirements.txt
├── .env                    ← API keys (never commit this)
└── README.md
```

---

## 🧠 System Architecture

### Machine Learning
- **Algorithm:** Random Forest Classifier (150 trees, max depth 12)
- **Target:** `Need_Maintenance` (binary: YES/NO)
- **Features:** 17 engineered features including date-derived fields
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Vehicle Health Index (VHI)
Weighted formula:
```
VHI = 0.30 × MileageHealth
    + 0.15 × BrakeScore
    + 0.15 × TireScore
    + 0.15 × BatteryScore
    + 0.10 × MaintenanceScore
    + 0.10 × ServiceScore
    + 0.05 × AccidentScore
```

### Remaining Useful Life (RUL)
Health-based approximate estimation:
```
RUL = (VHI / 100) × (300,000 − OdometerReading)
```
> ⚠️ This is NOT sensor/IoT-based RUL. It is a health-score approximation.

### SHAP Explainability
- TreeExplainer (optimized for Random Forest)
- Local explanations per vehicle
- Global feature importance chart
- Human-readable explanation text

### Rule-Based Diagnostics
Subsystems analyzed:
- Brake System
- Tire System  
- Battery System
- Maintenance Record
- Mileage / Odometer

### AI Assistant
- Primary: **Groq** (llama3-70b-8192) — fast & free
- Fallback: **OpenAI** (gpt-3.5-turbo)
- Async with timeout and retry handling
- Context-aware: uses vehicle analysis results

---

## 🖥️ Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 Dashboard | Platform overview & usage guide |
| 🔍 Vehicle Prediction | Input form + full analysis |
| 🧠 Explainable AI | SHAP waterfall + feature importance |
| 📈 RUL Analysis | Remaining Useful Life gauges |
| 🔧 Diagnostics | Subsystem risk breakdown |
| 🤖 AI Assistant | Conversational diagnostics chatbot |
| 📉 System Insights | Model metrics, ROC curve, confusion matrix |

---

## ⚙️ Configuration

### Environment Variables (`.env`)

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key (free at console.groq.com) | Optional |
| `OPENAI_API_KEY` | OpenAI API key | Optional |
| `GEMINI_API_KEY` | Google Gemini API key | Optional |

---

## 🔧 Troubleshooting

**Models not loading:**
Delete the `models/` folder and restart — the system will retrain.

**SHAP errors:**
SHAP explainer is auto-rebuilt if the cache is corrupted.

**AI assistant not working:**
Ensure your API key is in `.env` and restart the app.

**Port already in use:**
Change `server_port=7860` in `app.py` to another port.

---

## 📊 Dataset

- **Source:** `data/vehicle_maintenance_data.csv`
- **Size:** 50,000 records
- **Target:** `Need_Maintenance` (0/1)
- **Split:** 80% train / 20% test

---

## 📄 License

MIT License — Free to use and modify.
