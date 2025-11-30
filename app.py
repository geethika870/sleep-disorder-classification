import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42

# ------------------- App Layout -------------------

st.set_page_config(page_title="Sleep Disorder Classification", layout="wide")

st.sidebar.title("🧠 Model Controls")
page = st.sidebar.radio("Select Page", [
    "📤 Upload Data",
    "🤖 Train Model",
    "💾 Save/Load Model",
    "📊 Interpretability"
])

# ------------------- PAGE 1: DATA UPLOAD -------------------

if page == "📤 Upload Data":
    st.title("📤 Upload Sleep Dataset")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ Dataset Uploaded Successfully!")
        st.dataframe(df.head())

# ------------------- PAGE 2: MODEL TRAINING -------------------

elif page == "🤖 Train Model":
    st.title("🤖 Train Classification Model")

    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()

        # ✅ Encode categorical features
        label_encoders = {}
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        st.session_state.label_encoders = label_encoders
        st.session_state.feature_order = df.drop("Sleep Disorder", axis=1).columns.tolist()

        # ✅ Split X and y
        X = df.drop("Sleep Disorder", axis=1)
        y = df["Sleep Disorder"]

        # ✅ Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        st.session_state.scaler = scaler

        # ✅ Select model
        model_choice = st.sidebar.selectbox("Choose Model", [
            "Random Forest", "SVM", "KNN", "Logistic Regression", "Gradient Boosting"
        ])

        # ✅ Initialize model
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=200, random_state=SEED)
        elif model_choice == "SVM":
            model = SVC(probability=True, random_state=SEED)
        elif model_choice == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=300, random_state=SEED)
        else:
            model = GradientBoostingClassifier(random_state=SEED)

        # ✅ Train
        with st.spinner("Training Model..."):
            model.fit(X_scaled, y)

        # ✅ Save in session
        st.session_state.best_model = model
        st.success(f"✅ {model_choice} Model Trained!")

# ------------------- PAGE 3: SAVE / LOAD (OPTIMIZED) -------------------

elif page == "💾 Save/Load Model":
    st.title("💾 Save or Load Model")

    if "best_model" not in st.session_state:
        st.warning("Train model first!")
    else:
        if st.button("💾 Save Model"):
            start = time.time()
            pickle.dump(st.session_state.best_model, open("sleep_model.pkl", "wb"))
            pickle.dump(st.session_state.scaler, open("scaler.pkl", "wb"))
            pickle.dump(st.session_state.label_encoders, open("encoders.pkl", "wb"))
            st.success(f"✅ Model Saved in {round(time.time() - start, 2)} sec!")

        if st.button("📥 Load Model"):
            try:
                st.session_state.best_model = pickle.load(open("sleep_model.pkl", "rb"))
                st.session_state.scaler = pickle.load(open("scaler.pkl", "rb"))
                st.session_state.label_encoders = pickle.load(open("encoders.pkl", "rb"))
                st.success("✅ Model Loaded Successfully!")
            except:
                st.error("❌ No saved model found!")

# ------------------- PAGE 4: FEATURE IMPORTANCE (FIXED) -------------------

elif page == "📊 Interpretability":
    st.title("📊 Feature Importance")

    if "best_model" not in st.session_state:
        st.warning("Train or load model first!")
    elif "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()
        model = st.session_state.best_model
        scaler = st.session_state.scaler
        encoders = st.session_state.label_encoders
        order = st.session_state.feature_order

        # ✅ Encode categorical again safely
        for col, le in encoders.items():
            if col in df.columns and col != "Sleep Disorder":
                df[col] = df[col].map(lambda x: x if str(x) in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))

        X = df.drop("Sleep Disorder", axis=1)
        y = df["Sleep Disorder"]

        for f in order:
            if f not in X.columns:
                X[f] = 0

        X = X[order].astype(float)
        X_scaled = scaler.transform(X)

        # ✅ Importance
        result = permutation_importance(model, X_scaled, y, n_repeats=10, random_state=SEED)
        idx = result.importances_mean.argsort()[::-1]

        # ✅ Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=result.importances_mean[idx], y=np.array(order)[idx], ax=ax)
        st.pyplot(fig)

