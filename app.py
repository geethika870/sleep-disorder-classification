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
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
st.sidebar.title("⚙ Navigation")
page = st.sidebar.radio("Go to:", ["📂 Upload Dataset", "🚀 Train Models", "🔮 Predict Disorder", "📊 Interpretability"])

# -------------------
# FIXED LGBMWrapper CLASS
# -------------------
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


# -------------------
# SAVE / LOAD MODEL
# -------------------
def save_model(best_model, scaler, label_encoders, feature_order):
    try:
        with open("best_model.pkl", "wb") as f:
            pickle.dump((best_model, scaler, label_encoders, feature_order), f)
        return True
    except Exception as e:
        st.error(f"Failed to save model: {e}")
        return False


def load_model_file():
    if os.path.exists("best_model.pkl"):
        try:
            with open("best_model.pkl", "rb") as f:
                return pickle.load(f)
        except Exception:
            return None, None, None, None
    return None, None, None, None


# -------------------
# 📂 UPLOAD DATASET
# -------------------
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "Person ID" in df.columns:
            df.drop("Person ID", axis=1, inplace=True)

        if "Blood Pressure" in df.columns:
            try:
                df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
                df.drop("Blood Pressure", axis=1, inplace=True)
            except:
                st.warning("⚠ Could not parse Blood Pressure properly.")

        st.session_state.df = df
        st.success("✅ Dataset uploaded!")
        st.dataframe(df)


# -------------------
# 🚀 TRAIN MODELS
# -------------------
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare Models")

    if "df" not in st.session_state:
        st.warning("⚠ Upload dataset first!")
    else:
        df = st.session_state.df.copy()

        if "Sleep Disorder" not in df.columns:
            st.error("⚠ Dataset must contain Sleep Disorder column")
        else:
            label_encoders = {}

            for col in df.select_dtypes(include="object").columns:
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

            X = df.drop("Sleep Disorder", axis=1)
            y = df["Sleep Disorder"]

            try:
                smt = SMOTETomek(random_state=SEED)
                X_res, y_res = smt.fit_resample(X, y)
            except:
                X_res, y_res = X, y

            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2, random_state=SEED
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            st.info("⏳ Training models...")

            models = {
                "SVM": SVC(C=1, kernel="rbf", probability=True),
                "Random Forest": RandomForestClassifier(n_estimators=300),
                "LightGBM": LGBMWrapper(LGBMClassifier(n_estimators=300)),
                "CatBoost": CatBoostClassifier(iterations=300, verbose=0),
                "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False),
                "ANN": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500),
            }

            results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    preds = model.predict(X_test_scaled)
                    results[name] = accuracy_score(y_test, preds)
                except Exception as e:
                    st.warning(f"{name} failed: {e}")
                    results[name] = 0

            acc_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
            acc_df["Accuracy"] = (acc_df["Accuracy"] * 100).round(2)
            st.table(acc_df)

            best_name = acc_df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
            st.success(f"🏆 Best Model: {best_name}")

            st.session_state.best_model = models[best_name]
            st.session_state.scaler = scaler
            st.session_state.label_encoders = label_encoders
            st.session_state.feature_order = list(X.columns)

            cm = confusion_matrix(y_test, models[best_name].predict(X_test_scaled))
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            st.pyplot(fig)

            if st.button("💾 Save Model"):
                if save_model(models[best_name], scaler, label_encoders, list(X.columns)):
                    st.success("✅ Model Saved!")


# -------------------
# 🔮 PREDICT
# -------------------
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")

    if "best_model" not in st.session_state:
        loaded = load_model_file()
        if loaded[0]:
            st.session_state.best_model, st.session_state.scaler, st.session_state.label_encoders, st.session_state.feature_order = loaded
            st.success("Loaded saved model!")

    if "best_model" not in st.session_state:
        st.warning("⚠ Train/load a model first!")
    else:
        mode = st.radio("Mode", ["Manual Input", "Bulk"])

        if mode == "Manual Input":
            inputs = {}

            for feat in st.session_state.feature_order:
                if "gender" in feat.lower():
                    inputs[feat] = st.selectbox(feat, ["Male", "Female"])
                elif "age" in feat.lower():
                    inputs[feat] = st.slider(feat, 1, 100, 25)
                else:
                    inputs[feat] = st.number_input(feat, value=0.0)

            if st.button("Predict"):
                df = pd.DataFrame([inputs])
                for col, le in st.session_state.label_encoders.items():
                    if col in df:
                        df[col] = le.transform(df[col].astype(str))

                df = df[st.session_state.feature_order]
                scaled = st.session_state.scaler.transform(df)
                pred = st.session_state.best_model.predict(scaled)[0]

                le_target = st.session_state.label_encoders.get("Sleep Disorder")
                if le_target:
                    pred = le_target.inverse_transform([pred])[0]

                st.success(f"🩺 Sleep Disorder: {pred}")

        else:
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                new_df = pd.read_csv(file)

                for col, le in st.session_state.label_encoders.items():
                    if col in new_df:
                        new_df[col] = new_df[col].astype(str)
                        new_df[col] = new_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        new_df[col] = le.transform(new_df[col])

                try:
                    new_df = new_df[st.session_state.feature_order]
                except:
                    st.error("Column mismatch!")

                scaled = st.session_state.scaler.transform(new_df)
                preds = st.session_state.best_model.predict(scaled)

                le_target = st.session_state.label_encoders.get("Sleep Disorder")
                preds = le_target.inverse_transform(preds)

                new_df["Predicted"] = preds
                st.dataframe(new_df)


# -------------------
# 📊 INTERPRETABILITY
# -------------------
elif page == "📊 Interpretability":
    st.title("📊 Model Interpretability")

    if "best_model" not in st.session_state:
        st.warning("⚠ Train model first!")
    else:
        df = st.session_state.df.copy()
        label_encoders = st.session_state.label_encoders
        feature_order = st.session_state.feature_order

        for col, le in label_encoders.items():
            if col in df:
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])

        X = df[feature_order]
        y = df["Sleep Disorder"]

        scaler = st.session_state.scaler
        X_scaled = scaler.transform(X)

        st.info("⏳ Computing permutation importance...")

        result = permutation_importance(
            st.session_state.best_model, X_scaled, y,
            n_repeats=10, random_state=SEED
        )

        sorted_idx = result.importances_mean.argsort()[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=result.importances_mean[sorted_idx],
            y=np.array(feature_order)[sorted_idx],
            ax=ax
        )
        ax.set_title("Feature Importance (Permutation)")
        st.pyplot(fig)

