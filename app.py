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
from lightgbm import LGBMClassifier  # fixed usage with :contentReference[oaicite:0]{index=0}
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
# FIXED Wrapper Class
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
# Save / Load
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
# Upload Dataset
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
            except Exception:
                st.warning("Could not parse Blood Pressure; leaving it unchanged.")

        st.session_state.df = df
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(df.head())

# -------------------
# Train Models
# -------------------
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare Models")

    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()

        if "Sleep Disorder" not in df.columns:
            st.error("CSV must contain 'Sleep Disorder' column!")
        else:
            label_encoders = {}
            for col in df.select_dtypes(include="object").columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

            X = df.drop("Sleep Disorder", axis=1)
            y = df["Sleep Disorder"]

            try:
                smt = SMOTETomek(random_state=SEED)
                X_res, y_res = smt.fit_resample(X, y)
            except:
                st.warning("SMOTETomek failed — using original data.")
                X_res, y_res = X, y

            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2,
                stratify=y_res if len(np.unique(y_res)) > 1 else None,
                random_state=SEED
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            st.info("⏳ Training models...")

            models = {
                "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
                "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
                "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED),
                "CatBoost": CatBoostClassifier(iterations=300, verbose=0, random_state=SEED),
                "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=SEED),
                "ANN": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED)
            }

            results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    results[name] = accuracy_score(y_test, model.predict(X_test_scaled))
                except Exception as e:
                    st.warning(f"{name} failed: {e}")
                    results[name] = 0

            acc_df = pd.DataFrame(list(results.items()), columns=["Model","Accuracy"])
            acc_df["Accuracy"] = (acc_df["Accuracy"] * 100).round(2)
            st.table(acc_df)

            best_model_name = acc_df.loc[acc_df["Accuracy"].idxmax(), "Model"]
            st.success(f"🏆 Best Model: {best_model_name}")

            st.session_state.best_model = models[best_model_name]
            st.session_state.scaler = scaler
            st.session_state.label_encoders = label_encoders
            st.session_state.feature_order = list(X.columns)

            try:
                cm = confusion_matrix(y_test, models[best_model_name].predict(X_test_scaled))
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                st.pyplot(fig)
            except:
                st.text("Confusion Matrix:")
                st.write(cm)

            if st.button("💾 Save Best Model"):
                if save_model(models[best_model_name], scaler, label_encoders, list(X.columns)):
                    st.success("✅ Model saved!")

# -------------------
# Predict Disorder
# -------------------
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")

    if "best_model" not in st.session_state:
        bm, sc, le, fo = load_model_file()
        if bm:
            st.session_state.best_model, st.session_state.scaler, st.session_state.label_encoders, st.session_state.feature_order = bm, sc, le, fo
            st.success("Loaded saved model.")

    if "best_model" not in st.session_state:
        st.warning("Train or load model first!")
    else:
        mode = st.radio("Prediction Mode", ["Manual Input", "Bulk Prediction"])

        if mode == "Manual Input":
            inputs = {}
            for feat in st.session_state.feature_order:
                inputs[feat] = st.number_input(feat, value=0.0)

            if st.button("🔮 Predict"):
                user_data = pd.DataFrame([inputs])

                for col, le in st.session_state.label_encoders.items():
                    if col in user_data.columns:
                        val = str(user_data[col].iloc[0])
                        if val not in le.classes_:
                            le.classes_ = np.append(le.classes_, val)
                        user_data[col] = le.transform(user_data[col].astype(str))

                user_data = user_data[st.session_state.feature_order]
                scaled = st.session_state.scaler.transform(user_data.astype(float))
                pred_num = st.session_state.best_model.predict(scaled)[0]

                target_encoder = st.session_state.label_encoders.get("Sleep Disorder")
                pred_label = target_encoder.inverse_transform([int(pred_num)])[0] if target_encoder else pred_num

                st.success(f"🩺 Prediction: {pred_label}")

        else:
            file = st.file_uploader("Upload CSV without 'Sleep Disorder'", type=["csv"])
            if file:
                new_df = pd.read_csv(file)
                for col, le in st.session_state.label_encoders.items():
                    if col in new_df.columns:
                        new_df[col] = new_df[col].apply(lambda x: x if str(x) in le.classes_ else le.classes_[0])
                        new_df[col] = le.transform(new_df[col].astype(str))

                new_df = new_df[st.session_state.feature_order]
                scaled = st.session_state.scaler.transform(new_df.astype(float))
                preds = st.session_state.best_model.predict(scaled)

                target_encoder = st.session_state.label_encoders.get("Sleep Disorder")
                preds_labels = target_encoder.inverse_transform(preds.astype(int)) if target_encoder else preds

                new_df["Predicted Disorder"] = preds_labels
                st.dataframe(new_df.head())

                out = pickle.dumps(new_df)
                st.download_button("⬇ Download Predictions", out, "sleep_predictions.csv")

# -------------------
# Interpretability
# -------------------
elif page == "📊 Interpretability":
    st.title("📊 Feature Importance")

    if "best_model" not in st.session_state or "df" not in st.session_state:
        st.warning("Upload & train model first!")
    else:
        X = st.session_state.df[st.session_state.feature_order]
        y = st.session_state.df["Sleep Disorder"]
        le_target = st.session_state.label_encoders.get("Sleep Disorder")
        y_encoded = le_target.transform(y.astype(str)) if le_target else y

        X_scaled = st.session_state.scaler.transform(X.astype(float))

        result = permutation_importance(st.session_state.best_model, X_scaled, y_encoded, n_repeats=10, random_state=SEED)

        sorted_idx = result.importances_mean.argsort()[::-1]
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x=result.importances_mean[sorted_idx], y=np.array(st.session_state.feature_order)[sorted_idx], ax=ax)
        st.pyplot(fig)
