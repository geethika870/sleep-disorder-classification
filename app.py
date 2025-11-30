import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, random
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
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(
    page_title="Sleep Disorder Classification",
    layout="wide"
)

# -------------------
# FIXED LGBM WRAPPER
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
# SAVE & LOAD MODEL
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
        except:
            return None, None, None, None
    return None, None, None, None

# -------------------
# SIDEBAR NAVIGATION
# -------------------
st.sidebar.title("⚙ Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📂 Upload Dataset", "🚀 Train Models", "🔮 Predict Disorder"]
)

# HOME
if page == "🏠 Home":
    st.title("😴 Sleep Disorder Classification & Prediction")
    st.write("""
    Upload your dataset → Train ML Models → Predict Sleep Disorder → Download CSV Results.
    """)

# UPLOAD DATASET
elif page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        # clean blood pressure
        if "Blood Pressure" in df.columns:
            try:
                df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
                df.drop("Blood Pressure", axis=1, inplace=True)
            except:
                st.warning("Could not parse blood pressure column.")

        if "Person ID" in df.columns:
            df.drop("Person ID", axis=1, inplace=True)

        st.session_state.df = df
        st.success("Dataset uploaded successfully!")
        st.dataframe(df.head())

# TRAIN MODELS
elif page == "🚀 Train Models":
    st.title("🚀 Train & Compare ML Models")

    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()

        if "Sleep Disorder" not in df.columns:
            st.error("Dataset must contain 'Sleep Disorder' target column.")
        else:
            label_encoders = {}
            for col in df.select_dtypes(include="object").columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le

            X = df.drop("Sleep Disorder", axis=1)
            y = df["Sleep Disorder"]

            # handle imbalance
            try:
                smt = SMOTETomek(random_state=SEED)
                X_res, y_res = smt.fit_resample(X, y)
            except:
                X_res, y_res = X, y

            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2, random_state=SEED, stratify=y_res
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            st.info("Training models...")

            models = {
                "SVM": SVC(probability=True),
                "Random Forest": RandomForestClassifier(n_estimators=300),
                "LightGBM": LGBMWrapper(LGBMClassifier(n_estimators=300)),
                "CatBoost": CatBoostClassifier(iterations=300, verbose=0),
                "XGBoost": XGBClassifier(eval_metric="mlogloss"),
                "ANN": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
            }

            results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    results[name] = accuracy_score(y_test, preds)
                except Exception as e:
                    st.warning(f"{name} failed: {e}")
                    results[name] = 0.0

            acc_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
            acc_df["Accuracy"] = (acc_df["Accuracy"] * 100).round(2)
            st.table(acc_df)

            best_model_name = acc_df.iloc[acc_df["Accuracy"].idxmax()]["Model"]
            st.success(f"🏆 Best Model: {best_model_name}")

            best_model = models[best_model_name]

            st.session_state.best_model = best_model
            st.session_state.scaler = scaler
            st.session_state.label_encoders = label_encoders
            st.session_state.feature_order = list(X.columns)

            # Confusion matrix
            try:
                cm = confusion_matrix(y_test, best_model.predict(X_test))
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
            except:
                pass

            if st.button("💾 Save Best Model"):
                if save_model(best_model, scaler, label_encoders, list(X.columns)):
                    st.success("Model saved successfully!")

# PREDICT DISORDER
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")

    if "best_model" not in st.session_state:
        loaded = load_model_file()
        if loaded[0] is not None:
            st.session_state.best_model, st.session_state.scaler, st.session_state.label_encoders, st.session_state.feature_order = loaded
            st.success("Model loaded from file.")

    if "best_model" not in st.session_state:
        st.warning("Train/load a model first!")
    else:
        mode = st.radio("Prediction Mode", ["Manual Input", "Bulk CSV"])

        # MANUAL
        if mode == "Manual Input":
            inputs = {}
            for feat in st.session_state.feature_order:
                if "age" in feat.lower():
                    inputs[feat] = st.slider(feat, 1, 100, 25)
                else:
                    inputs[feat] = st.number_input(feat, value=0.0)

            if st.button("Predict"):
                user_df = pd.DataFrame([inputs])

                for col, encoder in st.session_state.label_encoders.items():
                    if col in user_df.columns:
                        user_df[col] = encoder.transform(user_df[col].astype(str))

                user_df = user_df[st.session_state.feature_order]
                scaled = st.session_state.scaler.transform(user_df)

                pred = st.session_state.best_model.predict(scaled)[0]

                target_encoder = st.session_state.label_encoders["Sleep Disorder"]
                label = target_encoder.inverse_transform([pred])[0]

                st.success(f"Predicted Sleep Disorder: {label}")

        # BULK CSV
        else:
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                new_df = pd.read_csv(file)

                # apply encoders
                for col, encoder in st.session_state.label_encoders.items():
                    if col in new_df.columns:
                        new_df[col] = new_df[col].astype(str)
                        missing = set(new_df[col]) - set(encoder.classes_)
                        if missing:
                            encoder.classes_ = np.append(encoder.classes_, list(missing))
                        new_df[col] = encoder.transform(new_df[col])

                new_df = new_df[st.session_state.feature_order]
                scaled = st.session_state.scaler.transform(new_df)

                preds = st.session_state.best_model.predict(scaled)

                target_encoder = st.session_state.label_encoders["Sleep Disorder"]
                new_df["Predicted Sleep Disorder"] = target_encoder.inverse_transform(preds)

                st.dataframe(new_df)

                # DOWNLOAD FILE
                csv = new_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇ Download Predictions CSV", csv, "predictions.csv")

