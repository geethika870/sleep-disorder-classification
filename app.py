import streamlit as st
import pandas as pd
import numpy as np
import pickle, joblib, os, random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from sklearn.inspection import permutation_importance
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
st.sidebar.title("⚙ Navigation")
page = st.sidebar.radio("Go to:", ["📂 Upload Dataset", "🚀 Train Models", "🔮 Predict Disorder", "📊 Interpretability"])

def save_model(best_model, scaler, label_encoders, feature_order):
    with open("best_model.pkl", "wb") as f:
        pickle.dump((best_model, scaler, label_encoders, feature_order), f)

def load_model():
    if os.path.exists("best_model.pkl"):
        return pickle.load(open("best_model.pkl","rb"))
    return None, None, None, None

# Upload Dataset
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        if "Person ID" in df.columns:
            df = df.drop("Person ID", axis=1)

        if "Blood Pressure" in df.columns:
            bp = df["Blood Pressure"].str.split("/", expand=True).astype(int)
            df["Systolic_BP"], df["Diastolic_BP"] = bp[0], bp[1]
            df = df.drop("Blood Pressure", axis=1)

        st.session_state.df = df
        st.success("✅ Dataset Uploaded")
        st.dataframe(df.head())

# Train Models
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare Models")
    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()
        if "Sleep Disorder" not in df.columns:
            st.error("Target column 'Sleep Disorder' missing!")
        else:
            encoders = {}
            for col in df.select_dtypes(include="object").columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

            X = df.drop("Sleep Disorder", axis=1)
            y = df["Sleep Disorder"]

            smt = SMOTETomek(random_state=SEED)
            X, y = smt.fit_resample(X, y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=SEED
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            models = {
                "SVM": SVC(C=1, kernel="rbf", probability=True, random_state=SEED),
                "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
                "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED),
                "CatBoost": CatBoostClassifier(iterations=300, verbose=0, random_state=SEED),
                "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=SEED),
                "ANN": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=400, random_state=SEED)
            }

            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                results[name] = accuracy_score(y_test, model.predict(X_test)) * 100

            acc_df = pd.DataFrame({
                "Model": list(results.keys()),
                "Accuracy (%)": np.round(list(results.values()),2)
            })

            st.table(acc_df)

            best = acc_df.loc[acc_df["Accuracy (%)"].idxmax()]
            st.success(f"🏆 Best Model: {best['Model']} → {best['Accuracy (%)']}%")

            st.session_state.best_model = models[best['Model']]
            st.session_state.scaler = scaler
            st.session_state.encoders = encoders
            st.session_state.feature_order = list(X.columns)

            save_model(st.session_state.best_model, scaler, encoders, st.session_state.feature_order)
            st.info("✅ Model auto-saved to disk")

            cm = confusion_matrix(y_test, st.session_state.best_model.predict(X_test))
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            st.pyplot(fig)

# Prediction
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")
    if "best_model" not in st.session_state:
        bm, sc, en, fo = load_model()
        if bm:
            st.session_state.best_model, st.session_state.scaler, st.session_state.encoders, st.session_state.feature_order = bm, sc, en, fo
            st.success("✅ Loaded saved model")

    if "best_model" not in st.session_state:
        st.warning("Train or upload model first!")
    else:
        mode = st.radio("Select Mode", ["Manual", "Bulk"])
        if mode == "Manual":
            user_input = {}
            for col in st.session_state.feature_order:
                user_input[col] = st.number_input(col, value=0.0)

            if st.button("Predict"):
                df = pd.DataFrame([user_input])
                df = df[st.session_state.feature_order]
                scaled = st.session_state.scaler.transform(df.astype(float))
                pred = st.session_state.best_model.predict(scaled)[0]
                st.success(f"🩺 Prediction: {pred}")

        else:
            file = st.file_uploader("Upload test CSV", type=["csv"])
            if file:
                df = pd.read_csv(file)

                if "Blood Pressure" in df.columns:
                    bp = df["Blood Pressure"].str.split("/", expand=True).astype(int)
                    df["Systolic_BP"], df["Diastolic_BP"] = bp[0], bp[1]
                    df = df.drop("Blood Pressure", axis=1)

                for col, le in st.session_state.encoders.items():
                    if col in df.columns:
                        df[col] = le.transform(df[col].astype(str))

                df = df[st.session_state.feature_order]
                scaled = st.session_state.scaler.transform(df.astype(float))
                preds = st.session_state.best_model.predict(scaled)
                df["Prediction"] = preds
                st.dataframe(df.head())

# Interpretability
elif page == "📊 Interpretability":
    st.title("📊 Feature Importance")

    if "df" not in st.session_state or "best_model" not in st.session_state:
        st.warning("Upload + train first!")
    else:
        df = st.session_state.df.copy()
        for col, le in st.session_state.encoders.items():
            if col in df.columns and df[col].dtype == "object":
                df[col] = df[col].apply(lambda x: x if str(x) in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))

        X = df[st.session_state.feature_order]
        y = df["Sleep Disorder"]
        X_scaled = st.session_state.scaler.transform(X.astype(float))

        imp = permutation_importance(st.session_state.best_model, X_scaled, y, n_repeats=8, random_state=SEED)
        idx = imp.importances_mean.argsort()[::-1]

        fig, ax = plt.subplots()
        ax.barh(np.array(st.session_state.feature_order)[idx], imp.importances_mean[idx])
        st.pyplot(fig)
        st.success("✅ Importance calculated")
