import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, random, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
page = st.sidebar.radio("Navigation", ["📤 Upload", "🚀 Train", "🔮 Predict", "📊 Interpret"])

# ---------------- LIGHTGBM WRAPPER FIX ----------------
class LGBMWrapper:
    def __init__(self, model):
        self.model = model
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    def __getattr__(self, attr):
        return getattr(self.model, attr)

# ---------------- UPLOAD DATA ----------------
if page == "📤 Upload":
    st.title("Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df = df.dropna(subset=["Sleep Disorder"])  # ✅ remove NaN target rows
        st.session_state.df = df
        st.success("✅ Uploaded & cleaned!")
        st.dataframe(df.head())

# ---------------- TRAIN 5 MODELS ----------------
elif page == "🚀 Train":
    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()

        # encode categoricals
        encoders = {}
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        X = df.drop("Sleep Disorder", axis=1)
        y = df["Sleep Disorder"]

        # split + scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 5 models only ✅
        models = {
            "SVM": SVC(probability=True, random_state=SEED),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=SEED),
            "LightGBM": LGBMWrapper(LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED)),
            "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=SEED),
            "ANN": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=SEED)
        }

        results = []
        best_acc = 0
        best_model = None
        best_name = ""

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results.append([name, acc])

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_name = name

        # display accuracies
        res_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
        res_df["Accuracy"] = (res_df["Accuracy"] * 100).round(2)
        st.table(res_df.sort_values(by="Accuracy", ascending=False))

        st.success(f"🏆 Best Model: {best_name} ({res_df['Accuracy'].max()}%)")

        # save best
        st.session_state.best_model = best_model
        st.session_state.best_acc = best_acc
        st.session_state.scaler = scaler
        st.session_state.label_encoders = encoders
        st.session_state.feature_order = X.columns.tolist()

        pickle.dump(st.session_state, open("best_model.pkl", "wb"))  # ✅ auto-save
        st.info("✅ Best model saved automatically!")

# ---------------- BULK PREDICTION ----------------
elif page == "🔮 Predict":
    if "best_model" not in st.session_state:
        st.warning("Train or load model first!")
    else:
        file = st.file_uploader("Upload CSV without target", type="csv")
        if file:
            new_df = pd.read_csv(file)

            # encode categoricals safely
            for col, le in st.session_state.label_encoders.items():
                if col in new_df.columns:
                    new_df[col] = new_df[col].apply(lambda x: x if str(x) in le.classes_ else le.classes_[0])
                    new_df[col] = le.transform(new_df[col].astype(str))

            # add/align missing columns
            for f in st.session_state.feature_order:
                if f not in new_df.columns:
                    new_df[f] = 0

            new_df = new_df[st.session_state.feature_order]
            X_scaled = st.session_state.scaler.transform(new_df.astype(float))
            preds = st.session_state.best_model.predict(X_scaled)

            # decode target
            target_le = st.session_state.label_encoders.get("Sleep Disorder")
            decoded = target_le.inverse_transform(preds.astype(int)) if target_le else preds

            new_df["Predicted Sleep Disorder"] = decoded
            st.success("✅ Predictions done!")
            st.dataframe(new_df.head())

            csv = new_df.to_csv(index=False).encode()
            st.download_button("📥 Download Predictions", csv, "predictions.csv")

# ---------------- FEATURE IMPORTANCE ----------------
elif page == "📊 Interpret":
    if "best_model" not in st.session_state:
        st.warning("Train/Load model first!")
    elif "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()
        X = df.drop("Sleep Disorder", axis=1)[st.session_state.feature_order]
        y = df["Sleep Disorder"]

        # encode again if needed
        for col, le in st.session_state.label_encoders.items():
            if col in X.columns:
                X[col] = le.transform(X[col].astype(str))

        X_scaled = st.session_state.scaler.transform(X.astype(float))

        try:
            imp = permutation_importance(st.session_state.best_model, X_scaled, y, n_repeats=5, random_state=SEED)
            importance = imp.importances_mean
        except:
            importance = st.session_state.best_model.feature_importances_

        # plot
        plt.figure()
        plt.barh(st.session_state.feature_order, importance)
        plt.gca().invert_yaxis()
        st.pyplot(plt)
