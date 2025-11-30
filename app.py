import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, random, time
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTETomek
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# ---------------- Config ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
st.sidebar.title("⚙ Navigation")
page = st.sidebar.radio("Go to:", ["📂 Upload Dataset", "🚀 Train Models", "🔮 Predict", "📊 Interpretability"])

# ---------------- Helpers ----------------
MODEL_PATH = "best_model.pkl"

def save_model(best_model, scaler, encoders, feature_list):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((best_model, scaler, encoders, feature_list), f)

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None, None, None, None

def safe_label_encode_series(s):
    le = LabelEncoder()
    return le.fit_transform(s.astype(str)), le

# ---------------- Upload ----------------
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        # keep rows where target exists; don't drop valid features
        if "Sleep Disorder" in df.columns:
            df = df.dropna(subset=["Sleep Disorder"])
        # parse BP if present
        if "Blood Pressure" in df.columns:
            try:
                df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(float)
                df.drop("Blood Pressure", axis=1, inplace=True)
            except Exception:
                st.warning("Could not parse 'Blood Pressure' column; left as-is.")
        st.session_state.df = df
        st.success("✅ Dataset uploaded")
        st.dataframe(df.head())

# ---------------- Train ----------------
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare Models (5)")
    if "df" not in st.session_state:
        st.warning("Upload dataset first.")
    else:
        df = st.session_state.df.copy()
        if "Sleep Disorder" not in df.columns:
            st.error("Dataset must contain 'Sleep Disorder' column.")
        else:
            # Encode target (label encoder saved)
            target_le = LabelEncoder()
            y = target_le.fit_transform(df["Sleep Disorder"].astype(str))
            st.session_state.target_le = target_le

            # Prepare features:
            X_raw = df.drop(columns=["Sleep Disorder"])
            # One-hot encode categorical features for trees / general reliability
            X = pd.get_dummies(X_raw, drop_first=True)

            # Fill NaNs in features with 0
            X = X.fillna(0)

            # Balance classes if possible
            try:
                smt = SMOTETomek(random_state=SEED)
                X_res, y_res = smt.fit_resample(X, y)
            except Exception:
                st.warning("SMOTETomek failed or not available — using original data.")
                X_res, y_res = X, y

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2, stratify=y_res, random_state=SEED
            )

            # Scaler for non-tree models
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Define 5 models (no wrapper) — LightGBM tuned
            models = {
                "SVM": SVC(C=1.0, kernel="rbf", probability=True, random_state=SEED),
                "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=20, random_state=SEED),
                "LightGBM": LGBMClassifier(n_estimators=600, learning_rate=0.03, max_depth=20, num_leaves=50,
                                           subsample=0.8, colsample_bytree=0.8, random_state=SEED),
                "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.04, max_depth=8, eval_metric="mlogloss", use_label_encoder=False, random_state=SEED),
                "ANN": MLPClassifier(hidden_layer_sizes=(200,100), max_iter=500, random_state=SEED)
            }

            st.info("⏳ Training models...")

            results = {}
            for name, model in models.items():
                try:
                    # Trees: use raw one-hot X; others use scaled
                    if name in ["LightGBM", "Random Forest", "XGBoost"]:
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                    else:
                        model.fit(X_train_scaled, y_train)
                        preds = model.predict(X_test_scaled)

                    results[name] = accuracy_score(y_test, preds)
                except Exception as e:
                    st.warning(f"{name} failed: {e}")
                    results[name] = 0.0

            acc_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
            acc_df["Accuracy (%)"] = (acc_df["Accuracy"] * 100).round(4)
            st.table(acc_df[["Model", "Accuracy (%)"]])

            best_idx = acc_df["Accuracy"].idxmax()
            best_name = acc_df.loc[best_idx, "Model"]
            st.success(f"🏆 Best Model: {best_name} ({acc_df.loc[best_idx,'Accuracy (%)']}%)")

            # Save best model and pipeline components
            best_model = models[best_name]
            st.session_state.best_model = best_model
            st.session_state.scaler = scaler
            st.session_state.feature_list = X.columns.tolist()
            st.session_state.encoders = {}  # we used get_dummies for features; target encoder stored separately
            st.session_state.target_le = target_le

            try:
                save_model(best_model, scaler, st.session_state.encoders, st.session_state.feature_list)
                st.info("💾 Best model saved to disk (best_model.pkl).")
            except Exception as e:
                st.warning(f"Could not auto-save model: {e}")

            # Confusion matrix (attach colorbar correctly)
            try:
                if best_name in ["LightGBM", "Random Forest", "XGBoost"]:
                    cm = confusion_matrix(y_test, best_model.predict(X_test))
                else:
                    cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
                fig, ax = plt.subplots()
                im = ax.imshow(cm, aspect="auto")
                plt.colorbar(im, ax=ax)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, cm[i, j], ha="center", va="center")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            except Exception:
                pass

# ---------------- Predict ----------------
elif page == "🔮 Predict":
    st.title("🔮 Predict Sleep Disorder")
    # try to load saved model if session doesn't have one
    if "best_model" not in st.session_state:
        bm, sc, enc, feat = load_model()
        if bm is not None:
            st.session_state.best_model, st.session_state.scaler, st.session_state.encoders, st.session_state.feature_list = bm, sc, enc, feat
            st.success("Loaded saved model from disk")

    if "best_model" not in st.session_state:
        st.warning("Train or load a model first")
    else:
        mode = st.radio("Mode", ["Manual Input", "Bulk CSV"])

        if mode == "Manual Input":
            st.info("Manual input will expect the same feature columns as training (one-hot columns).")
            # For simplicity, present a text area to paste a single-row CSV compatible with training features
            raw = st.text_area("Paste a single-row CSV (header + one row) matching training features (or use Bulk CSV). Example:\nFeature1,Feature2,...\nval1,val2,...")
            if st.button("Predict from text"):
                try:
                    tmp = pd.read_csv(pd.compat.StringIO(raw))
                    # one-hot alignment
                    tmp = pd.get_dummies(tmp).reindex(columns=st.session_state.feature_list, fill_value=0)
                    # scale if model expects scaled input? Our pipeline uses scaler for non-tree only; we will try both
                    try:
                        pred = st.session_state.best_model.predict(st.session_state.scaler.transform(tmp))
                    except Exception:
                        pred = st.session_state.best_model.predict(tmp.values)
                    if "target_le" in st.session_state and st.session_state.target_le is not None:
                        pred_label = st.session_state.target_le.inverse_transform(pred.astype(int))
                        st.success(f"Prediction: {pred_label[0]}")
                    else:
                        st.success(f"Prediction: {pred[0]}")
                except Exception as e:
                    st.error(f"Could not predict from input: {e}")

        else:
            file = st.file_uploader("Upload CSV without 'Sleep Disorder' (bulk)", type=["csv"])
            if file:
                new_df = pd.read_csv(file)
                # parse BP if present
                if "Blood Pressure" in new_df.columns:
                    try:
                        new_df[["Systolic_BP", "Diastolic_BP"]] = new_df["Blood Pressure"].str.split("/", expand=True).astype(float)
                        new_df.drop("Blood Pressure", axis=1, inplace=True)
                    except Exception:
                        pass

                new_df = new_df.fillna(0)
                new_X = pd.get_dummies(new_df).reindex(columns=st.session_state.feature_list, fill_value=0)

                # predict using model (try scaled first then raw)
                try:
                    X_for_pred = st.session_state.scaler.transform(new_X)
                    preds = st.session_state.best_model.predict(X_for_pred)
                except Exception:
                    preds = st.session_state.best_model.predict(new_X.values)

                # decode target if encoder present
                if "target_le" in st.session_state and st.session_state.target_le is not None:
                    try:
                        preds_labels = st.session_state.target_le.inverse_transform(preds.astype(int))
                    except Exception:
                        preds_labels = preds
                else:
                    preds_labels = preds

                new_df["Predicted_Sleep_Disorder"] = preds_labels
                st.dataframe(new_df.head())
                csv_out = new_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", csv_out, "predictions.csv", "text/csv")

# ---------------- Interpret ----------------
elif page == "📊 Interpretability":
    st.title("📊 Feature Importance")
    if "best_model" not in st.session_state:
        st.warning("Train/load a model first")
    else:
        try:
            df = st.session_state.df.copy()
            X_full = pd.get_dummies(df.drop(columns=["Sleep Disorder"])).reindex(columns=st.session_state.feature_list, fill_value=0)
            y_full = LabelEncoder().fit_transform(df["Sleep Disorder"].astype(str))
            X_full = X_full.fillna(0)
            X_scaled_full = st.session_state.scaler.transform(X_full)
            result = permutation_importance(st.session_state.best_model, X_scaled_full, y_full, n_repeats=8, random_state=SEED, scoring="accuracy")
            idx = result.importances_mean.argsort()[::-1]
            feat_names = np.array(st.session_state.feature_list)[idx]
            importances = result.importances_mean[idx]
            fig, ax = plt.subplots(figsize=(10,6))
            ax.barh(feat_names, importances)
            ax.invert_yaxis()
            ax.set_xlabel("Mean importance")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Interpretability failed: {e}")
