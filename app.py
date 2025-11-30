import streamlit as st
import pandas as pd
import numpy as np
import pickle, joblib, os, random
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Page Config
st.set_page_config(page_title="SleepSense Pro", layout="wide")

# ---------------- Fixed LGBM Wrapper ----------------
class LGBMWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __getattr__(self, attr):
        return getattr(self.model, attr)

# ---------------- Save & Load ----------------
def save_model(best_model, scaler, label_encoders, feature_order):
    with open("best_model.pkl", "wb") as f:
        pickle.dump((best_model, scaler, label_encoders, feature_order), f)

def load_model_file():
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            return pickle.load(f)
    return None, None, None, None

# ---------------- Sidebar ----------------
st.sidebar.title("🌙 SleepSense Pro")
page = st.sidebar.radio("Navigate:", ["🏠 Home", "📂 Upload Dataset", "🚀 Train Models", "🔮 Predict", "📊 Analysis"])

# ---------------- Home Page UI ----------------
if page == "🏠 Home":
    st.markdown("""
    <style>
    body {background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);}
    .title {font-size: 55px; font-weight: bold; text-align: center; color: white;}
    .sub {font-size: 20px; text-align: center; color: #dcdcdc;}
    .card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">😴 SleepSense Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">AI Powered Sleep Disorder Prediction | Medical Grade ML</div><br>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="card">🧠 Improves Memory & Focus</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">💓 Detects Early Health Risks</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card">📉 80% Apnea Undiagnosed</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        🔍 Why Sleep AI?<br>
        • 70M Americans suffer yearly<br>
        • $411B economic burden<br>
        • 98%+ detection possible using ML
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        ⚡ What You Can Do<br>
        1. Upload CSV<br>
        2. Train 6+ Medical ML Models<br>
        3. Predict for Manual or Bulk<br>
        4. See feature importance insights
        </div>
        """, unsafe_allow_html=True)

# ---------------- Upload Dataset ----------------
elif page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.success("✅ Dataset uploaded!")
        st.dataframe(df.head())

# ---------------- Train Models ----------------
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare ML Models")
    if "df" not in st.session_state:
        st.error("Upload a dataset first!")
    else:
        df = st.session_state.df.copy()

        if "Sleep Disorder" not in df.columns:
            st.error("Target column 'Sleep Disorder' missing!")
        else:
            le_target = LabelEncoder()
            df["Sleep Disorder"] = le_target.fit_transform(df["Sleep Disorder"].astype(str))

            label_encoders = {"Sleep Disorder": le_target}
            for col in df.select_dtypes(include="object").columns:
                if col != "Sleep Disorder":
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    label_encoders[col] = le

            X = df.drop("Sleep Disorder", axis=1)
            y = df["Sleep Disorder"]

            smt = SMOTETomek(random_state=SEED)
            X_res, y_res = smt.fit_resample(X, y)

            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=SEED)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
                "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
                "LightGBM": LGBMWrapper(LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED)),
                "CatBoost": CatBoostClassifier(iterations=300, verbose=0, random_state=SEED),
                "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=SEED),
                "ANN": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=SEED)
            }

            results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    results[name] = accuracy_score(y_test, model.predict(X_test_scaled))
                except Exception:
                    results[name] = 0

            acc_df = pd.DataFrame(results.items(), columns=["Model","Accuracy"])
            acc_df["Accuracy"] = (acc_df["Accuracy"]*100).round(2)
            st.table(acc_df)

            best_model_name = acc_df.loc[acc_df["Accuracy"].idxmax(),"Model"]
            st.success(f"🏆 Best Model: {best_model_name}")

            st.session_state.best_model = models[best_model_name]
            st.session_state.scaler = scaler
            st.session_state.label_encoders = label_encoders
            st.session_state.feature_order = list(X.columns)

            if st.button("💾 Save Model"):
                save_model(st.session_state.best_model, scaler, label_encoders, list(X.columns))
                st.success("✅ Model saved!")

# ---------------- Prediction ----------------
elif page == "🔮 Predict":
    st.title("🔮 Sleep Disorder Prediction")

    if "best_model" not in st.session_state:
        bm, sc, le, fo = load_model_file()
        if bm:
            st.session_state.best_model, st.session_state.scaler, st.session_state.label_encoders, st.session_state.feature_order = bm, sc, le, fo
            st.success("✅ Loaded saved model!")

    if "best_model" not in st.session_state:
        st.error("Train or upload a saved model first!")
    else:
        mode = st.radio("Predict Mode", ["Manual","Bulk"])
        if mode == "Manual":
            inputs = {}
            for f in st.session_state.feature_order:
                inputs[f] = st.number_input(f, 0.0)
            if st.button("Predict"):
                user_df = pd.DataFrame([inputs])
                scaled = st.session_state.scaler.transform(user_df)
                pred = st.session_state.label_encoders["Sleep Disorder"].inverse_transform([int(st.session_state.best_model.predict(scaled)[0])])[0]
                st.success(f"🩺 Prediction: {pred}")

        else:
            file = st.file_uploader("Upload CSV",type=["csv"])
            if file:
                df_new = pd.read_csv(file)

                for col, le in st.session_state.label_encoders.items():
                    if col in df_new.columns:
                        df_new[col] = df_new[col].apply(lambda x: x if str(x) in le.classes_ else le.classes_[0])
                        df_new[col] = le.transform(df_new[col].astype(str))

                df_new = df_new[st.session_state.feature_order]
                scaled = st.session_state.scaler.transform(df_new)
                df_new["Prediction"] = st.session_state.label_encoders["Sleep Disorder"].inverse_transform(st.session_state.best_model.predict(scaled).astype(int))

                st.dataframe(df_new.head())

                # CSV Download
                csv = df_new.to_csv(index=False).encode('utf-8')
                st.download_button("⬇ Download Predictions", csv, "sleep_predictions.csv", "text/csv")

# ---------------- Feature Importance ----------------
elif page == "📊 Analysis":
    st.title("📊 Feature Importance")
    if "best_model" not in st.session_state:
        st.warning("Train model first!")
    else:
        df = st.session_state.df.copy()
        for col, le in st.session_state.label_encoders.items():
            if col in df.columns and col != "Sleep Disorder":
                df[col] = df[col].apply(lambda x: x if str(x) in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))
        X = df[st.session_state.feature_order]
        y = df["Sleep Disorder"]
        X_scaled = st.session_state.scaler.transform(X)
        result = permutation_importance(st.session_state.best_model, X_scaled, y, n_repeats=10, random_state=SEED)
        fi = pd.DataFrame({"Feature":X.columns,"Importance":result.importances_mean}).sort_values("Importance",ascending=False)
        st.bar_chart(fi.set_index("Feature"))
