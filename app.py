import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, random
from sklearn.model_selection import train_test_split, StratifiedKFold
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
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
st.sidebar.title("⚙ Navigation")
page = st.sidebar.radio("Go to:", ["📂 Upload Dataset", "🚀 Train Models", "🔮 Predict Disorder", "📊 Interpretability"])

# ✅ FIXED LightGBM Wrapper
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

def save_model(best_model, scaler, encoders, feature_order):
    with open("best_model.pkl", "wb") as f:
        pickle.dump((best_model, scaler, encoders, feature_order), f)

def load_model_file():
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            return pickle.load(f)
    return None, None, None, None

# ------------------- DATA UPLOAD -------------------
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        # Cleaning (kept minimal)
        if "Person ID" in df.columns:
            df.drop("Person ID", axis=1, inplace=True)
        if "Blood Pressure" in df.columns:
            try:
                df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
                df.drop("Blood Pressure", axis=1, inplace=True)
            except:
                pass

        st.session_state.df = df
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(df.head())

# ------------------- TRAIN MODELS -------------------
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare 5 Models")

    # ✅ FIX: Load uploaded dataset instead of file path
    if "df" not in st.session_state:
        st.error("❗ Upload dataset first!")
        st.stop()
    df = st.session_state.df.copy()

    if "Sleep Disorder" not in df.columns:
        st.error("❗ Target column `Sleep Disorder` missing in dataset!")
        st.stop()

    # ✅ FIX: Remove NaN in target to stop crash
    df = df.dropna(subset=["Sleep Disorder"])

    # Encode categoricals
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop("Sleep Disorder", axis=1)
    y = df["Sleep Disorder"]

    # Balance classes
    smt = SMOTETomek(random_state=SEED)
    X_res, y_res = smt.fit_resample(X, y)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    # 5 models only ✅
    models = {
        "SVM": SVC(probability=True, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=25, random_state=SEED),
        # 🔥 Tuned for higher accuracy
        "LightGBM": LGBMClassifier(n_estimators=1200, learning_rate=0.02, num_leaves=90, max_depth=25,
                                   subsample=0.85, colsample_bytree=0.85, random_state=SEED),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False,
                                 n_estimators=400, learning_rate=0.05, max_depth=20, random_state=SEED),
        "ANN": MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=600, random_state=SEED)
    }

    # K-Fold comparison for fairness
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    for name, model in models.items():
        scores=[]
        for train_i, test_i in skf.split(X_scaled, y_res):
            model.fit(X_scaled[train_i], y_res.iloc[train_i])
            pred = model.predict(X_scaled[test_i])
            scores.append(accuracy_score(y_res.iloc[test_i], pred)*100)
        results[name] = np.mean(scores).round(4)

    acc_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy (%)"])
    st.table(acc_df)

    best = acc_df.iloc[acc_df["Accuracy (%)"].idxmax()]["Model"]
    st.success(f"🏆 Best Model: {best} ({acc_df['Accuracy (%)'].max()}%)")

    st.session_state.best_model = models[best]
    st.session_state.scaler = scaler
    st.session_state.encoders = encoders
    st.session_state.feature_order = list(X.columns)

    if st.button("💾 Save Best Model"):
        save_model(models[best], scaler, encoders, list(X.columns))
        st.success("✅ Best model saved!")

# ------------------- PREDICTION -------------------
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")

    # Load if saved
    if "best_model" not in st.session_state:
        b, s, e, f = load_model_file()
        if b:
            st.session_state.best_model = b
            st.session_state.scaler = s
            st.session_state.encoders = e
            st.session_state.feature_order = f
            st.success("Loaded model from disk!")

    if "best_model" not in st.session_state:
        st.error("❗ No model found. Train or upload first!")
        st.stop()

    mode = st.radio("Prediction Mode", ["Manual", "Bulk"])

    if mode == "Manual":
        inputs={}
        for feat in st.session_state.feature_order:
            inputs[feat]=st.number_input(feat, value=0.0)
        
        if st.button("🔮 Predict"):
            user_df = pd.DataFrame([inputs])
            user_df = user_df[st.session_state.feature_order]
            scaled = st.session_state.scaler.transform(user_df.astype(float))
            pred = st.session_state.best_model.predict(scaled)[0]
            le = st.session_state.encoders.get("Sleep Disorder")
            label = le.inverse_transform([int(pred)])[0] if le else pred
            st.success(f"🩺 Prediction: {label}")

    else:
        file = st.file_uploader("Upload Bulk CSV", type=["csv"])
        if file:
            new_df = pd.read_csv(file)
            for col,le in st.session_state.encoders.items():
                if col in new_df.columns:
                    new_df[col]=new_df[col].apply(lambda x: x if str(x) in le.classes_ else le.classes_[0])
                    new_df[col]=le.transform(new_df[col].astype(str))
            new_df=new_df[st.session_state.feature_order]
            scaled=st.session_state.scaler.transform(new_df.astype(float))
            preds=st.session_state.best_model.predict(scaled)
            le=st.session_state.encoders.get("Sleep Disorder")
            labels=le.inverse_transform(preds.astype(int)) if le else preds
            new_df["Prediction"]=labels
            st.dataframe(new_df.head())
            new_df.to_csv("predictions.csv", index=False)
            st.download_button("⬇ Download Predictions CSV","predictions.csv")

# ------------------- INTERPRETABILITY -------------------
elif page == "📊 Interpretability":
    st.title("📊 Feature Importance")
    if "best_model" not in st.session_state:
        st.error("❗ Train model first!")
        st.stop()
    df = st.session_state.df.copy().dropna()
    X=df[st.session_state.feature_order]
    y=df["Sleep Disorder"]
    if y.dtype=="object":
        le=st.session_state.encoders.get("Sleep Disorder")
        y=le.transform(y) if le else y
    X_scaled = st.session_state.scaler.transform(X.astype(float))
    result = permutation_importance(st.session_state.best_model, X_scaled, y, n_repeats=10, random_state=SEED)
    idx=result.importances_mean.argsort()[::-1]
    fig,ax=plt.subplots()
    ax.barh(np.array(st.session_state.feature_order)[idx], result.importances_mean[idx])
    st.pyplot(fig)

