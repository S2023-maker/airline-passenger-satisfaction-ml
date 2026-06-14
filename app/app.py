"""
✈️ Airline Passenger Satisfaction Predictor
Sprint 4: Deployment & MLOps
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Airline Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding: 1rem; }
    .stMetric { background: #f0f2f6; padding: 10px; border-radius: 10px; }
    .prediction-box-satisfied {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white; padding: 20px; border-radius: 15px;
        text-align: center; font-size: 24px; font-weight: bold;
    }
    .prediction-box-dissatisfied {
        background: linear-gradient(135deg, #dc3545, #fd7e14);
        color: white; padding: 20px; border-radius: 15px;
        text-align: center; font-size: 24px; font-weight: bold;
    }
    .sprint-badge {
        background: #2c3e50; color: white; padding: 5px 12px;
        border-radius: 20px; font-size: 13px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained pipeline from pickle."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model_pipeline.pkl')
    le_path    = os.path.join(os.path.dirname(__file__), '..', 'models', 'label_encoder.pkl')
    meta_path  = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_meta.pkl')

    if not os.path.exists(model_path):
        return None, None, None

    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    with open(le_path, 'rb') as f:
        le_y = pickle.load(f)
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return pipeline, le_y, meta

pipeline, le_y, meta = load_model()

# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Airplane_silhouette.svg/200px-Airplane_silhouette.svg.png", width=60)
st.sidebar.title("✈️ Airline ML Project")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home & Overview", "🔮 Predict Satisfaction", "📊 EDA Dashboard", "🔬 Model Insights", "📋 Sprint Summary"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**4-Sprint ML Methodology**")
sprints = {
    "Sprint 1": "Data Understanding & Preprocessing",
    "Sprint 2": "Model Building & Evaluation",
    "Sprint 3": "Optimization & Final Model",
    "Sprint 4": "Deployment & MLOps ← You are here",
}
for s, desc in sprints.items():
    color = "#28a745" if "here" in desc else "#6c757d"
    st.sidebar.markdown(f"<small style='color:{color}'>**{s}**: {desc}</small>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helper: Build input dataframe with feature engineering
# ─────────────────────────────────────────────
def build_input_df(inputs: dict) -> pd.DataFrame:
    """Apply same feature engineering as Sprint 3."""
    service_cols = [
        'departure_and_arrival_time_convenience', 'ease_of_online_booking',
        'check_in_service', 'online_boarding', 'gate_location',
        'on_board_service', 'seat_comfort', 'leg_room_service',
        'cleanliness', 'food_and_drink', 'in_flight_service',
        'in_flight_wifi_service', 'in_flight_entertainment', 'baggage_handling'
    ]
    row = inputs.copy()
    row['total_delay']       = row['departure_delay'] + row['arrival_delay']
    row['delay_ratio']       = row['arrival_delay'] / (row['departure_delay'] + 1)
    row['avg_service_rating']= np.mean([row[c] for c in service_cols])
    row['inflight_experience']= np.mean([row['seat_comfort'], row['in_flight_entertainment'],
                                          row['in_flight_service'], row['in_flight_wifi_service']])
    row['ground_experience'] = np.mean([row['check_in_service'], row['gate_location'],
                                         row['baggage_handling'], row['ease_of_online_booking']])
    row['is_long_haul']      = int(row['flight_distance'] > 1000)
    # Drop 'id' if present
    return pd.DataFrame([row])

# ─────────────────────────────────────────────
# Page: Home
# ─────────────────────────────────────────────
if page == "🏠 Home & Overview":
    st.title("✈️ Airline Passenger Satisfaction Predictor")
    st.markdown("##### A complete ML solution built using a structured 4-Sprint methodology")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Project Type", "Classification")
    with col2:
        st.metric("📦 Total Sprints", "4")
    with col3:
        if meta:
            st.metric("🤖 Best Model", meta['model_name'].split('(')[0].strip())
        else:
            st.metric("🤖 Best Model", "Train first")
    with col4:
        if meta:
            st.metric("✅ Test F1 Score", f"{meta['test_f1']:.3f}")
        else:
            st.metric("✅ Test F1 Score", "—")

    st.markdown("---")
    st.subheader("📋 Project Structure")

    sprint_data = {
        "Sprint": ["Sprint 1", "Sprint 2", "Sprint 3", "Sprint 4"],
        "Focus Area": [
            "Data Understanding & Preprocessing",
            "Model Building & Evaluation",
            "Optimization & Final Model",
            "Deployment & MLOps"
        ],
        "Key Deliverables": [
            "Clean dataset, EDA plots, Preprocessing pipeline",
            "6 trained models, Evaluation metrics, ROC-AUC curves",
            "Hyperparameter tuning, Feature engineering, Saved model (.pkl)",
            "Streamlit UI, End-to-end pipeline, Documentation"
        ],
        "Status": ["✅ Complete", "✅ Complete", "✅ Complete", "✅ Live"]
    }
    st.dataframe(pd.DataFrame(sprint_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🎯 Problem Statement")
    st.info("""
    **Goal:** Predict whether an airline passenger will be **Satisfied** or **Neutral/Dissatisfied**
    based on their flight experience, demographics, and service ratings.

    **Dataset Features:** Demographics (age, gender), Flight details (distance, class, delays),
    Service ratings (14 attributes rated 0–5), and Customer type.

    **Business Value:** Helps airlines identify key drivers of passenger satisfaction and
    proactively improve service quality.
    """)

# ─────────────────────────────────────────────
# Page: Predict
# ─────────────────────────────────────────────
elif page == "🔮 Predict Satisfaction":
    st.title("🔮 Passenger Satisfaction Predictor")
    st.markdown("Fill in the passenger details to get a satisfaction prediction.")

    if pipeline is None:
        st.error("⚠️ Model not found! Please run Sprint 3 notebook first to train and save the model.")
        st.stop()

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("👤 Passenger Profile")
        gender        = st.selectbox("Gender", ["Male", "Female"])
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        age           = st.slider("Age", 7, 85, 35)
        type_of_travel= st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        travel_class  = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
        flight_distance = st.slider("Flight Distance (km)", 50, 5000, 1000)
        departure_delay = st.slider("Departure Delay (min)", 0, 300, 0)
        arrival_delay   = st.slider("Arrival Delay (min)", 0, 300, 0)

    with col_right:
        st.subheader("⭐ Service Ratings (0 = Worst, 5 = Best)")
        wifi        = st.slider("In-flight WiFi Service",           0, 5, 3)
        time_conv   = st.slider("Departure/Arrival Time Convenience", 0, 5, 3)
        online_book = st.slider("Ease of Online Booking",           0, 5, 3)
        gate_loc    = st.slider("Gate Location",                    0, 5, 3)
        food        = st.slider("Food and Drink",                   0, 5, 3)
        online_brd  = st.slider("Online Boarding",                  0, 5, 3)
        seat        = st.slider("Seat Comfort",                     0, 5, 3)
        entertain   = st.slider("In-flight Entertainment",          0, 5, 3)
        onboard_svc = st.slider("On-board Service",                 0, 5, 3)
        leg_room    = st.slider("Leg Room Service",                 0, 5, 3)
        baggage     = st.slider("Baggage Handling",                 0, 5, 3)
        checkin     = st.slider("Check-in Service",                 0, 5, 3)
        inflight    = st.slider("In-flight Service",                0, 5, 3)
        cleanliness = st.slider("Cleanliness",                      0, 5, 3)

    st.markdown("---")
    if st.button("🚀 Predict Satisfaction", type="primary", use_container_width=True):
        inputs = {
            'id': 0,
            'gender': gender,
            'customer_type': customer_type,
            'age': age,
            'type_of_travel': type_of_travel,
            'class': travel_class,
            'flight_distance': flight_distance,
            'inflight_wifi_service': wifi,
            'departure_and_arrival_time_convenience': time_conv,
            'ease_of_online_booking': online_book,
            'gate_location': gate_loc,
            'food_and_drink': food,
            'online_boarding': online_brd,
            'seat_comfort': seat,
            'in_flight_entertainment': entertain,
            'on_board_service': onboard_svc,
            'leg_room_service': leg_room,
            'baggage_handling': baggage,
            'check_in_service': checkin,
            'in_flight_service': inflight,
            'cleanliness': cleanliness,
            'departure_delay': departure_delay,
            'arrival_delay': arrival_delay,
            'in_flight_wifi_service': wifi,
        }

        input_df = build_input_df(inputs)

        # Keep only expected columns
        if meta:
            expected_cols = meta['cat_cols'] + meta['num_cols']
            input_df = input_df[[c for c in expected_cols if c in input_df.columns]]

        prediction = pipeline.predict(input_df)[0]
        proba      = pipeline.predict_proba(input_df)[0]

        result_label = le_y.inverse_transform([prediction])[0]
        conf = proba[prediction] * 100

        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if result_label == "satisfied":
                st.markdown(f"""
                <div class="prediction-box-satisfied">
                    😊 SATISFIED<br>
                    <small>Confidence: {conf:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box-dissatisfied">
                    😞 NEUTRAL / DISSATISFIED<br>
                    <small>Confidence: {conf:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("😊 Satisfied Probability",       f"{proba[1]*100:.1f}%")
        with col_b:
            st.metric("😞 Dissatisfied Probability",    f"{proba[0]*100:.1f}%")

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(6, 2))
        classes = le_y.classes_
        colors  = ['#dc3545' if c != 'satisfied' else '#28a745' for c in classes]
        ax.barh(classes, proba, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Prediction Confidence")
        for i, v in enumerate(proba):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        plt.tight_layout()
        st.pyplot(fig)

# ─────────────────────────────────────────────
# Page: EDA Dashboard
# ─────────────────────────────────────────────
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")
    st.markdown("Sprint 1: Data Understanding & Preprocessing results")

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'airline_passenger_satisfaction.csv')
    if not os.path.exists(data_path):
        st.warning("⚠️ Dataset not found. Place `airline_passenger_satisfaction.csv` in the `data/` folder.")
        st.stop()

    @st.cache_data
    def load_data(path):
        df = pd.read_csv(path)
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy='median')
        df["Arrival Delay"] = imp.fit_transform(df[["Arrival Delay"]])
        df.drop_duplicates(inplace=True)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
        if 'unnamed:_0' in df.columns:
            df.drop('unnamed:_0', axis=1, inplace=True)
        return df

    df = load_data(data_path)

    # Dataset overview
    st.subheader("📋 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",    f"{df.shape[0]:,}")
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())
    c4.metric("Duplicates",    df.duplicated().sum())

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Target", "Demographics", "Service Ratings", "Correlations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            vc = df['satisfaction'].value_counts()
            ax.bar(vc.index, vc.values, color=['#FF6B6B', '#4ECDC4'], edgecolor='white')
            ax.set_title("Satisfaction Distribution")
            for i, v in enumerate(vc.values):
                ax.text(i, v + 100, str(v), ha='center', fontweight='bold')
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            vc.plot.pie(ax=ax, autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'], startangle=90)
            ax.set_title("Proportion")
            ax.set_ylabel('')
            st.pyplot(fig)

    with tab2:
        cat_cols_eda = ['gender', 'customer_type', 'type_of_travel', 'class']
        for col in cat_cols_eda:
            if col in df.columns:
                fig, ax = plt.subplots(figsize=(8, 3))
                ct = pd.crosstab(df[col], df['satisfaction'], normalize='index') * 100
                ct.plot(kind='barh', ax=ax, color=['#FF6B6B', '#4ECDC4'])
                ax.set_title(f"{col} vs Satisfaction (%)")
                ax.set_xlabel("Percentage")
                ax.legend(loc='lower right')
                st.pyplot(fig)

    with tab3:
        service_cols = [
            'departure_and_arrival_time_convenience', 'ease_of_online_booking',
            'check_in_service', 'online_boarding', 'gate_location',
            'on_board_service', 'seat_comfort', 'leg_room_service',
            'cleanliness', 'food_and_drink', 'in_flight_service',
            'in_flight_wifi_service', 'in_flight_entertainment', 'baggage_handling'
        ]
        avg_ratings = {}
        for col in service_cols:
            if col in df.columns:
                avg_ratings[col.replace('_', ' ').title()] = df[col].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        cols_sorted = sorted(avg_ratings, key=avg_ratings.get, reverse=True)
        vals = [avg_ratings[c] for c in cols_sorted]
        bars = ax.barh(cols_sorted, vals, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(cols_sorted))))
        ax.set_xlabel("Average Rating (0–5)")
        ax.set_title("Average Service Ratings")
        ax.set_xlim(0, 5)
        for i, v in enumerate(vals):
            ax.text(v + 0.05, i, f"{v:.2f}", va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        num_data = df.select_dtypes(include=[np.number])
        if 'satisfaction' in df.columns:
            from sklearn.preprocessing import LabelEncoder
            le_tmp = LabelEncoder()
            num_data['satisfaction_enc'] = le_tmp.fit_transform(df['satisfaction'])
        corr = num_data.corr()
        fig, ax = plt.subplots(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', linewidths=0.5, annot_kws={'size': 6}, ax=ax)
        ax.set_title("Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig)

# ─────────────────────────────────────────────
# Page: Model Insights
# ─────────────────────────────────────────────
elif page == "🔬 Model Insights":
    st.title("🔬 Model Insights & Comparison")

    if pipeline is None:
        st.error("⚠️ Model not found. Run Sprint 3 notebook first.")
        st.stop()

    # Model comparison table
    st.subheader("📊 Sprint 2 — Model Comparison")
    comparison_data = {
        "Model"      : ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVM", "Naïve Bayes"],
        "Train Acc"  : ["~87%", "~100%", "~99%", "~97%", "~93%", "~80%"],
        "Test Acc"   : ["~87%", "~94%", "~96%", "~93%", "~93%", "~80%"],
        "F1 Score"   : ["~0.87", "~0.94", "~0.96", "~0.93", "~0.93", "~0.80"],
        "Overfitting": ["No", "Yes", "Slight", "Slight", "No", "No"],
        "Remarks"    : [
            "Good baseline, misses non-linear patterns",
            "Overfits training data",
            "Best overall balance",
            "Strong performer",
            "Computationally expensive",
            "Low accuracy on complex patterns"
        ]
    }
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🏆 Sprint 3 — Final Optimized Model")
    if meta:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🤖 Model", meta['model_name'].split('(')[0])
        c2.metric("✅ Test Accuracy", f"{meta['test_accuracy']:.3f}")
        c3.metric("🎯 Test F1 Score", f"{meta['test_f1']:.3f}")
        c4.metric("📦 Pickle File", "final_model_pipeline.pkl")

    st.markdown("---")
    st.subheader("🔧 Hyperparameter Tuning")
    tuning_info = {
        "Model"              : ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        "Tuning Method"      : ["GridSearchCV", "RandomizedSearchCV", "GridSearchCV"],
        "CV Folds"           : [5, 5, 5],
        "Scoring Metric"     : ["F1", "F1", "F1"],
        "Key Parameters Tuned": [
            "n_estimators, max_depth, min_samples_split",
            "n_estimators, learning_rate, max_depth, subsample",
            "C, solver"
        ]
    }
    st.dataframe(pd.DataFrame(tuning_info), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("⚙️ Feature Engineering")
    fe_data = {
        "Feature"           : ["total_delay", "delay_ratio", "avg_service_rating",
                                "inflight_experience", "ground_experience", "is_long_haul"],
        "Description"       : [
            "departure_delay + arrival_delay",
            "arrival_delay / (departure_delay + 1)",
            "Mean of all 14 service ratings",
            "Mean of seat comfort, entertainment, WiFi, in-flight service",
            "Mean of check-in, gate, baggage, online booking",
            "1 if flight_distance > 1000 km, else 0"
        ],
        "Rationale"         : [
            "Combined delay impact",
            "Delay pattern indicator",
            "Overall service quality",
            "Core in-flight experience",
            "Pre/post-flight experience",
            "Long haul vs short haul behavior"
        ]
    }
    st.dataframe(pd.DataFrame(fe_data), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Page: Sprint Summary
# ─────────────────────────────────────────────
elif page == "📋 Sprint Summary":
    st.title("📋 Sprint Deliverables Summary")
    st.markdown("Complete overview of all 4 sprints and their deliverables.")

    sprints_detail = [
        {
            "sprint": "Sprint 1: Data Understanding & Preprocessing",
            "color": "#e8f5e9",
            "deliverables": [
                "✔ Dataset loaded and validated (103,904 rows × 24 cols)",
                "✔ Missing values imputed (Arrival Delay — median)",
                "✔ Duplicates removed",
                "✔ Outliers capped using IQR (departure/arrival delays)",
                "✔ EDA — Univariate, Bivariate, Multivariate analysis",
                "✔ Feature encoding (OHE for categorical, StandardScaler for numeric)",
                "✔ 80-20 stratified train-test split",
                "✔ Preprocessing pipeline saved"
            ]
        },
        {
            "sprint": "Sprint 2: Model Building & Evaluation",
            "color": "#e3f2fd",
            "deliverables": [
                "✔ Baseline: Logistic Regression (F1 ~0.87)",
                "✔ Decision Tree Classifier",
                "✔ Random Forest Classifier",
                "✔ Gradient Boosting Classifier",
                "✔ Support Vector Machine (RBF kernel)",
                "✔ Naïve Bayes (Gaussian)",
                "✔ Confusion matrices for all models",
                "✔ ROC-AUC curves comparison",
                "✔ 5-Fold cross-validation",
                "✔ Feature importance (Random Forest)"
            ]
        },
        {
            "sprint": "Sprint 3: Optimization & Final Model",
            "color": "#fff3e0",
            "deliverables": [
                "✔ 6 new features engineered (total_delay, avg_service_rating, etc.)",
                "✔ Feature selection via importance & correlation",
                "✔ GridSearchCV for Random Forest & Logistic Regression",
                "✔ RandomizedSearchCV for Gradient Boosting",
                "✔ Final model pipeline (Preprocessor + Best Model)",
                "✔ Learning curve analysis",
                "✔ Model serialized with pickle (.pkl)",
                "✔ Label encoder saved"
            ]
        },
        {
            "sprint": "Sprint 4: Deployment & MLOps",
            "color": "#fce4ec",
            "deliverables": [
                "✔ Streamlit web application deployed",
                "✔ Interactive prediction UI with all input features",
                "✔ EDA Dashboard integrated",
                "✔ Model insights and comparison page",
                "✔ End-to-end ML pipeline",
                "✔ Probability confidence display",
                "✔ Structured project folder (data/, models/, notebooks/, app/)",
                "✔ Documentation complete"
            ]
        }
    ]

    for s in sprints_detail:
        with st.expander(s["sprint"], expanded=True):
            cols = st.columns(2)
            half = len(s["deliverables"]) // 2
            with cols[0]:
                for item in s["deliverables"][:half]:
                    st.markdown(item)
            with cols[1]:
                for item in s["deliverables"][half:]:
                    st.markdown(item)

    st.markdown("---")
    st.subheader("📁 Project Structure")
    st.code("""
airline_ml/
├── data/
│   ├── airline_passenger_satisfaction.csv   ← Raw dataset
│   └── airline_cleaned.csv                  ← Sprint 1 output
├── notebooks/
│   ├── sprint1_data_preprocessing.ipynb
│   ├── sprint2_model_building.ipynb
│   ├── sprint3_optimization.ipynb
│   └── sprint4_deployment.ipynb
├── models/
│   ├── final_model_pipeline.pkl             ← Final trained pipeline
│   ├── label_encoder.pkl
│   └── feature_meta.pkl
├── app/
│   └── app.py                               ← This Streamlit app
└── requirements.txt
    """, language="bash")

    st.info("**Run the app:** `streamlit run app/app.py`")
