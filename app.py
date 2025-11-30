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


# =============== FIXED LGBMWrapper ===============
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


# ================= SAVE / LOAD ==================
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


# =================== UPLOAD PAGE ==================
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
                st.warning("Could not parse Blood Pressure column")

        st.session_state.df = df
        st.success("Dataset uploaded successfully!")
        st.dataframe(df.head())


# =================== TRAIN MODELS ==================
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare Models")

    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()

        if "Sleep Disorder" not in df.columns:
            st.error("Dataset must contain Sleep Disorder column")
        else:
            label_encoders = {}

            # Encode object columns
            for col in df.select_dtypes(include="object").columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

            X = df.drop("Sleep Disorder", axis=1)
            y = df["Sleep Disorder"]

            # Handle imbalance
            try:
                smt = SMOTETomek(random_state=SEED)
                X_res, y_res = smt.fit_resample(X, y)
            except:
                st.warning("SMOTETomek failed → using original data")
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
                "ANN (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500),
            }

            results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    preds = model.predict(X_test_scaled)
                    results[name] = accuracy_score(y_test, preds)
                except Exception as e:
                    st.warning(f"{name} failed → {e}")
                    results[name] = 0.0

            acc_df = pd.DataFrame(results.items(), columns=["Model", "Accuracy"])
            acc_df["Accuracy"] = (acc_df["Accuracy"] * 100).round(2)

            st.table(acc_df)

            best_model_name = acc_df.iloc[acc_df["Accuracy"].idxmax()]["Model"]
            st.success(f"🏆 Best Model: {best_model_name}")

            st.session_state.best_model = models[best_model_name]
            st.session_state.scaler = scaler
            st.session_state.label_encoders = label_encoders
            st.session_state.feature_order = list(X.columns)

            # Confusion Matrix
            try:
                cm = confusion_matrix(y_test, models[best_model_name].predict(X_test_scaled))
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
            except:
                pass

            if st.button("💾 Save Best Model"):
                if save_model(models[best_model_name], scaler, label_encoders, X.columns):
                    st.success("Model Saved!")


# =================== PREDICT PAGE ==================
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")

    if "best_model" not in st.session_state:
        loaded = load_model_file()
        if loaded[0] is not None:
            st.session_state.best_model, st.session_state.scaler, st.session_state.label_encoders, st.session_state.feature_order = loaded
            st.success("Loaded saved model")

    if "best_model" not in st.session_state:
        st.warning("Train or load model first")
    else:
        mode = st.radio("Prediction Mode", ["Manual", "Bulk"])

        if mode == "Manual":

            inputs = {}
            for feat in st.session_state.feature_order:
                if "gender" in feat.lower():
                    inputs[feat] = st.selectbox(feat, ["Male", "Female"])
                elif "age" in feat.lower():
                    inputs[feat] = st.slider(feat, 1, 100, 25)
                else:
                    inputs[feat] = st.number_input(feat, value=0.0)

            if st.button("Predict"):
                df_new = pd.DataFrame([inputs])

                # Encode
                for col, le in st.session_state.label_encoders.items():
                    if col in df_new:
                        df_new[col] = le.transform(df_new[col].astype(str))

                df_new = df_new[st.session_state.feature_order]
                df_scaled = st.session_state.scaler.transform(df_new)

                pred = st.session_state.best_model.predict(df_scaled)[0]

                target_enc = st.session_state.label_encoders.get("Sleep Disorder")
                pred_label = target_enc.inverse_transform([pred])[0] if target_enc else pred

                st.success(f"Predicted: {pred_label}")

        else:
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                new_df = pd.read_csv(file)

                # Encode cols
                for col, le in st.session_state.label_encoders.items():
                    if col in new_df:
                        new_df[col] = new_df[col].astype(str).apply(
                            lambda x: x if x in le.classes_ else le.classes_[0]
                        )
                        new_df[col] = le.transform(new_df[col])

                new_df = new_df[st.session_state.feature_order]
                df_scaled = st.session_state.scaler.transform(new_df)

                preds = st.session_state.best_model.predict(df_scaled)

                target_enc = st.session_state.label_encoders.get("Sleep Disorder")
                new_df["Predicted_Sleep_Disorder"] = (
                    target_enc.inverse_transform(preds) if target_enc else preds
                )

                st.dataframe(new_df)
                st.download_button("Download Predictions CSV", new_df.to_csv(index=False), "predictions.csv")


# =================== INTERPRETABILITY ==================
elif page == "📊 Interpretability":
    st.title("📊 Model Interpretability")

    if "best_model" not in st.session_state:
        st.warning("Train a model first!")
    else:
        best_model = st.session_state.best_model
        df = st.session_state.df.copy()
        scaler = st.session_state.scaler
        feature_order = st.session_state.feature_order

        # Encode
        for col, le in st.session_state.label_encoders.items():
            if col in df:
                df[col] = le.transform(df[col].astype(str))

        X = df[feature_order]
        y = df["Sleep Disorder"]

        X_scaled = scaler.transform(X)

        st.info("⏳ Calculating permutation importance...")

        result = permutation_importance(best_model, X_scaled, y, n_repeats=10, random_state=SEED)

        importances = result.importances_mean
        sorted_idx = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances[sorted_idx], y=np.array(feature_order)[sorted_idx], ax=ax)
        ax.set_title("Permutation Feature Importance")
        st.pyplot(fig)

