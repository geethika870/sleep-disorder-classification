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
from sklearn.metrics import accuracy_score

SEED = 42

st.set_page_config(page_title="Sleep Disorder Classifier", layout="wide")

page = st.sidebar.radio("Navigation", [
    "📤 Upload Data",
    "🚀 Train & Compare Models",
    "💾 Save/Load Best Model",
    "🔮 Predict Disorder"
])

# ---------------- PAGE 1: UPLOAD DATA ----------------
if page == "📤 Upload Data":
    st.title("📤 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.success("✅ Dataset Uploaded!")
        st.dataframe(df.head())

# ---------------- PAGE 2: TRAIN ALL MODELS + PICK BEST ----------------
elif page == "🚀 Train & Compare Models":
    st.title("🚀 Training All Models & Comparing Accuracy")

    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()

        # Encode categorical columns
        encoders = {}
        for col in df.select_dtypes(include="object").columns:
            if col != "Sleep Disorder":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

        st.session_state.label_encoders = encoders
        st.session_state.feature_order = df.drop("Sleep Disorder", axis=1).columns.tolist()

        X = df.drop("Sleep Disorder", axis=1)
        y = df["Sleep Disorder"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        st.session_state.scaler = scaler

        # Define models
        models = {
            "SVM": SVC(probability=True, random_state=SEED),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(max_iter=300, random_state=SEED),
            "Gradient Boosting": GradientBoostingClassifier(random_state=SEED)
        }

        report = []
        best_acc = 0
        best_model = None

        progress = st.progress(0)
        total = len(models)

        for i, (name, model) in enumerate(models.items()):
            with st.spinner(f"Training {name}..."):
                model.fit(X_scaled, y)
                preds = model.predict(X_scaled)
                acc = accuracy_score(y, preds)
                report.append((name, acc))

                if acc > best_acc:
                    best_acc = acc
                    best_model = model

            progress.progress(int(((i+1)/total)*100))

        # Store best
        st.session_state.best_model = best_model
        st.session_state.best_acc = best_acc

        # Display comparison
        result_df = pd.DataFrame(report, columns=["Model", "Accuracy"])
        st.dataframe(result_df.sort_values(by="Accuracy", ascending=False))
        st.success(f"🏆 Best Model Selected with {round(best_acc*100, 2)}% Accuracy!")

# ---------------- PAGE 3: SAVE / LOAD BEST MODEL ----------------
elif page == "💾 Save/Load Best Model":
    st.title("💾 Save or Load the Best Model")

    if "best_model" not in st.session_state:
        st.warning("Train models first!")
    else:
        if st.button("💾 Save Best Model"):
            start = time.time()
            pickle.dump(st.session_state.best_model, open("best_sleep_model.pkl", "wb"))
            pickle.dump(st.session_state.scaler, open("scaler.pkl", "wb"))
            pickle.dump(st.session_state.label_encoders, open("encoders.pkl", "wb"))
            st.success(f"✅ Best Model Saved in {round(time.time() - start, 2)} sec!")

        if st.button("📥 Load Best Model"):
            try:
                st.session_state.best_model = pickle.load(open("best_sleep_model.pkl", "rb"))
                st.session_state.scaler = pickle.load(open("scaler.pkl", "rb"))
                st.session_state.label_encoders = pickle.load(open("encoders.pkl", "rb"))
                st.success("✅ Best Model Loaded!")
            except:
                st.error("❌ No saved model found!")

# ---------------- PAGE 4: SAFE PREDICTION ----------------
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")

    if "best_model" not in st.session_state:
        st.warning("Train or load best model first!")
    else:
        model = st.session_state.best_model
        scaler = st.session_state.scaler
        encoders = st.session_state.label_encoders
        order = st.session_state.feature_order

        upload = st.file_uploader("Upload CSV without Sleep Target", type=["csv"])
        if upload:
            new_df = pd.read_csv(upload)

            for col, le in encoders.items():
                if col in new_df.columns:
                    new_df[col] = new_df[col].map(lambda x: x if str(x) in le.classes_ else le.classes_[0])
                    new_df[col] = le.transform(new_df[col].astype(str))

            for f in order:
                if f not in new_df.columns:
                    new_df[f] = 0

            new_df = new_df[order].astype(float)
            X_scaled = scaler.transform(new_df)

            out = model.predict(X_scaled)
            new_df["Predicted Disorder"] = out

            st.success("✅ Predictions Complete!")
            st.dataframe(new_df)

            new_df.to_csv("prediction_results.csv", index=False)
            with open("prediction_results.csv", "rb") as dl:
                st.download_button("📥 Download Predictions", dl, "Predictions.csv")

