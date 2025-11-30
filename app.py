# app.py
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
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
st.sidebar.title("⚙ Navigation")
page = st.sidebar.radio("Go to:", ["📂 Upload Dataset", "🚀 Train Models", "🔮 Predict Disorder", "📊 Interpretability"])

# -----------------------
# Optional libraries (import safely)
# -----------------------
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    from imblearn.combine import SMOTETomek
except Exception:
    SMOTETomek = None

# -----------------------
# Fixed LGBM wrapper
# -----------------------
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

# -----------------------
# save / load helpers
# -----------------------
MODEL_PATH = "best_model.pkl"

def save_model(best_model, scaler, label_encoders, feature_order):
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump((best_model, scaler, label_encoders, feature_order), f)
        return True
    except Exception as e:
        st.error(f"Failed to save model: {e}")
        return False

def load_model_file():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None, None, None, None
    return None, None, None, None

# -----------------------
# Upload dataset
# -----------------------
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    uploaded = st.file_uploader("Upload CSV (one-time for session)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            # convenience parsing
            if "Person ID" in df.columns:
                df.drop("Person ID", axis=1, inplace=True)
            if "Blood Pressure" in df.columns:
                try:
                    df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(float)
                    df.drop("Blood Pressure", axis=1, inplace=True)
                except Exception:
                    st.warning("Could not parse 'Blood Pressure' column; left unchanged.")
            st.session_state.df = df
            st.success("Dataset uploaded and stored in session.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    else:
        if "df" in st.session_state:
            st.info("Dataset already loaded in this session.")
            st.dataframe(st.session_state.df.head())

# -----------------------
# Train Models
# -----------------------
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare Models")
    if "df" not in st.session_state:
        st.warning("Upload dataset first under 'Upload Dataset'.")
    else:
        df = st.session_state.df.copy()
        if "Sleep Disorder" not in df.columns:
            st.error("Dataset must include a 'Sleep Disorder' column (target).")
        else:
            # encode categorical columns
            label_encoders = {}
            for col in df.select_dtypes(include=["object", "category"]).columns:
                le = LabelEncoder()
                df[col] = df[col].astype(str).fillna("NA")
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

            X = df.drop(columns=["Sleep Disorder"])
            y = df["Sleep Disorder"].values

            # handle imbalance safely
            if SMOTETomek is not None:
                try:
                    smt = SMOTETomek(random_state=SEED)
                    X_res, y_res = smt.fit_resample(X, y)
                except Exception:
                    st.warning("SMOTETomek failed — using original data.")
                    X_res, y_res = X, y
            else:
                X_res, y_res = X, y

            # safe stratify
            stratify_arg = y_res if len(np.unique(y_res)) > 1 else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_res, y_res, test_size=0.2, random_state=SEED, stratify=stratify_arg
                )
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_res, y_res, test_size=0.2, random_state=SEED
                )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            st.info("⏳ Training models... (this may take a while for heavy models)")

            # build models dict dynamically based on availability
            models = {
                "SVM": SVC(C=1, kernel="rbf", probability=True, random_state=SEED),
                "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED),
                "ANN": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED)
            }
            if XGBClassifier is not None:
                models["XGBoost"] = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=SEED)
            else:
                st.warning("xgboost not installed — skipping XGBoost.")

            if LGBMClassifier is not None:
                # wrap to ensure interface parity
                models["LightGBM"] = LGBMWrapper(LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED))
            else:
                st.info("lightgbm not installed — skipping LightGBM.")

            if CatBoostClassifier is not None:
                models["CatBoost"] = CatBoostClassifier(iterations=300, verbose=0, random_state=SEED)
            else:
                st.info("catboost not installed — skipping CatBoost.")

            results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    preds = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, preds)
                    results[name] = acc
                    st.write(f"{name} trained — accuracy: {acc:.4f}")
                except Exception as e:
                    st.warning(f"{name} failed: {e}")
                    results[name] = 0.0

            acc_df = pd.DataFrame([(k, v) for k, v in results.items()], columns=["Model", "Accuracy"])
            acc_df["Accuracy"] = (acc_df["Accuracy"] * 100).round(2)
            st.table(acc_df)

            best_model_name = acc_df.loc[acc_df["Accuracy"].idxmax(), "Model"]
            st.success(f"🏆 Best model: {best_model_name}")

            best_model = models[best_model_name]
            st.session_state.best_model = best_model
            st.session_state.scaler = scaler
            st.session_state.label_encoders = label_encoders
            st.session_state.feature_order = list(X.columns)

            # confusion matrix (use scaled test set)
            try:
                cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not compute confusion matrix: {e}")

            if st.button("💾 Save Best Model"):
                ok = save_model(best_model, scaler, label_encoders, st.session_state.feature_order)
                if ok:
                    st.success("Model saved to disk.")

# -----------------------
# Predict
# -----------------------
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")
    # try load model if not in session
    if "best_model" not in st.session_state:
        loaded = load_model_file()
        if loaded[0] is not None:
            st.session_state.best_model, st.session_state.scaler, st.session_state.label_encoders, st.session_state.feature_order = loaded
            st.success("Loaded model from file.")

    if "best_model" not in st.session_state:
        st.warning("Train or load a model first.")
    else:
        mode = st.radio("Prediction mode", ["Manual", "Bulk CSV"])
        feature_order = st.session_state.feature_order

        if mode == "Manual":
            st.markdown("Enter feature values (form generated from training features).")
            inputs = {}
            for feat in feature_order:
                # make friendly guesses
                if "age" in feat.lower():
                    inputs[feat] = st.slider(feat, 1, 100, 25)
                elif "gender" in feat.lower():
                    inputs[feat] = st.selectbox(feat, ["Male", "Female"])
                else:
                    inputs[feat] = st.number_input(feat, value=0.0)

            if st.button("Predict"):
                user_df = pd.DataFrame([inputs])
                # apply encoders where applicable
                for col, le in st.session_state.label_encoders.items():
                    if col in user_df.columns:
                        val = str(user_df.at[0, col])
                        if val not in le.classes_:
                            le.classes_ = np.append(le.classes_, val)
                        user_df[col] = le.transform(user_df[col].astype(str))

                # reorder and scale
                user_df = user_df[feature_order]
                X_user = user_df.astype(float)
                X_user_scaled = st.session_state.scaler.transform(X_user)
                pred_num = st.session_state.best_model.predict(X_user_scaled)[0]

                # decode target if encoder exists
                target_le = st.session_state.label_encoders.get("Sleep Disorder")
                try:
                    pred_label = target_le.inverse_transform([int(pred_num)])[0] if target_le is not None else pred_num
                except Exception:
                    pred_label = pred_num
                st.success(f"Predicted Sleep Disorder: {pred_label}")

        else:
            uploaded = st.file_uploader("Upload CSV for bulk prediction (no target column required)", type=["csv"])
            if uploaded is not None:
                new_df = pd.read_csv(uploaded)
                # optional BP parse
                if "Blood Pressure" in new_df.columns:
                    try:
                        new_df[["Systolic_BP", "Diastolic_BP"]] = new_df["Blood Pressure"].str.split("/", expand=True).astype(float)
                        new_df.drop("Blood Pressure", axis=1, inplace=True)
                    except Exception:
                        st.warning("Could not parse Blood Pressure in uploaded file.")

                # apply encoders
                for col, le in st.session_state.label_encoders.items():
                    if col in new_df.columns:
                        new_df[col] = new_df[col].astype(str)
                        missing = set(new_df[col]) - set(le.classes_)
                        if missing:
                            le.classes_ = np.append(le.classes_, list(missing))
                        new_df[col] = le.transform(new_df[col])

                # align columns
                try:
                    X_new = new_df[feature_order].astype(float)
                except Exception:
                    X_new = new_df.iloc[:, :len(feature_order)].astype(float)
                    X_new.columns = feature_order

                X_new_scaled = st.session_state.scaler.transform(X_new)
                preds = st.session_state.best_model.predict(X_new_scaled)

                target_le = st.session_state.label_encoders.get("Sleep Disorder")
                try:
                    preds_labels = target_le.inverse_transform(preds.astype(int)) if target_le is not None else preds
                except Exception:
                    preds_labels = preds

                out = new_df.copy()
                out["Predicted_Sleep_Disorder"] = preds_labels
                st.dataframe(out.head(50))
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download predictions CSV", csv, "predictions.csv")

# -----------------------
# Interpretability
# -----------------------
elif page == "📊 Interpretability":
    st.title("📊 Feature Importance (Permutation)")
    if "best_model" not in st.session_state or "df" not in st.session_state:
        st.warning("Train/load a model and upload dataset first.")
    else:
        df = st.session_state.df.copy()
        feature_order = st.session_state.feature_order
        label_encoders = st.session_state.label_encoders

        # encode dataset consistently
        for col, le in label_encoders.items():
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype(str)
                missing = set(df[col]) - set(le.classes_)
                if missing:
                    le.classes_ = np.append(le.classes_, list(missing))
                df[col] = le.transform(df[col])

        X = df[feature_order].astype(float)
        y = df["Sleep Disorder"]
        if label_encoders.get("Sleep Disorder") is not None:
            y = label_encoders["Sleep Disorder"].transform(y.astype(str))

        X_scaled = st.session_state.scaler.transform(X)

        st.info("⏳ Computing permutation importance (may take time)...")
        try:
            from sklearn.inspection import permutation_importance
            res = permutation_importance(st.session_state.best_model, X_scaled, y, n_repeats=10, random_state=SEED, n_jobs=1)
            idx = res.importances_mean.argsort()[::-1]
            fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(feature_order))))
            sns.barplot(x=res.importances_mean[idx], y=np.array(feature_order)[idx], ax=ax)
            ax.set_xlabel("Mean importance")
            st.pyplot(fig)
            st.success("Feature importance computed.")
        except Exception as e:
            st.error(f"Permutation importance failed: {e}")
