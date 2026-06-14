# ✈️ Airline Passenger Satisfaction — ML Project

A complete machine learning solution using a structured **4-Sprint methodology**.

## 📋 Project Overview

| Sprint | Focus | Key Output |
|--------|-------|-----------|
| Sprint 1 | Data Understanding & Preprocessing | Clean dataset, EDA report |
| Sprint 2 | Model Building & Evaluation | 6 trained models, metrics, ROC-AUC |
| Sprint 3 | Optimization & Final Model | Tuned model, saved `.pkl` |
| Sprint 4 | Deployment & MLOps | Streamlit web app |

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add dataset
Place `airline_passenger_satisfaction.csv` into the `data/` folder.

### 3. Run notebooks in order
```bash
jupyter notebook notebooks/sprint1_data_preprocessing.ipynb
jupyter notebook notebooks/sprint2_model_building.ipynb
jupyter notebook notebooks/sprint3_optimization.ipynb
jupyter notebook notebooks/sprint4_deployment.ipynb
```

### 4. Launch Streamlit app
```bash
streamlit run app/app.py
```

## 🤖 Models Trained (Sprint 2)
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (RBF)
- Naïve Bayes (Gaussian)

## ⚙️ Feature Engineering (Sprint 3)
- `total_delay` — combined departure + arrival delay
- `delay_ratio` — arrival/departure delay ratio
- `avg_service_rating` — mean of all 14 service ratings
- `inflight_experience` — WiFi + entertainment + in-flight service + comfort
- `ground_experience` — check-in + gate + baggage + online booking
- `is_long_haul` — binary flag for flights > 1000 km

## 📦 Model Serialization
The final pipeline (preprocessing + classifier) is saved using `pickle`:
```python
import pickle
with open('models/final_model_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)
prediction = pipeline.predict(X_new)
```

## 📁 Project Structure
```
airline_ml/
├── data/
├── notebooks/
│   ├── sprint1_data_preprocessing.ipynb
│   ├── sprint2_model_building.ipynb
│   ├── sprint3_optimization.ipynb
│   └── sprint4_deployment.ipynb
├── models/
│   ├── final_model_pipeline.pkl
│   ├── label_encoder.pkl
│   └── feature_meta.pkl
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

## 🎯 Target Variable
- `satisfied` → 1
- `neutral or dissatisfied` → 0
