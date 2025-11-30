# app.py
"""
Full Streamlit app — single file
- Sidebar navigation (Home, Upload, Train & Compare, Manual Predict, Bulk Predict, Interpretability)
- Robust preprocessing with OneHotEncoder compatibility (sparse_output vs sparse)
- Three models: ANN, ANN+GA (lightweight), Hybrid (ANN -> XGBoost)
- Saves best model + preprocessing to disk (best_model.pkl)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import random
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# optional imports
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    import shap
except Exception:
    shap = None

try:
    from imblearn.combine import SMOTETomek
except Exception:
    SMOTETomek = None

# -------------------------
# Config & CSS
# -------------------------
SEED = 42
MODEL_FILE = "best_model.pkl"
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Sleep Disorder Classifier", layout="wide")
st.markdown("""
    <style>
    .card {border-radius:12px; padding:12px; background: linear-gradient(180deg,#fff,#f4fbff); box-shadow:0 6px 18px rgba(8,61,119,0.06); border:1px solid rgba(11,72,107,0.04); }
    .metric {font-size:28px; font-weight:700; color:#083d77;}
    .label {color:#6b7885; font-size:13px;}
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Session state initialization
# -------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "preproc" not in st.session_state:
    st.session_state.preproc = None  # dict: {ohe, scaler, num_cols, cat_cols, target_le}
if "models" not in st.session_state:
    st.session_state.models = {}
if "best" not in st.session_state:
    st.session_state.best = {"name": None, "score": None}
if "saved" not in st.session_state:
    st.session_state.saved = False

# -------------------------
# OneHotEncoder compatibility wrapper
# -------------------------
def make_onehotencoder():
    from sklearn.preprocessing import OneHotEncoder
    try:
        # newer scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# -------------------------
# Save / Load helpers
# -------------------------
def save_to_disk(obj, path=MODEL_FILE):
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

def load_from_disk(path=MODEL_FILE):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

# -------------------------
# Preprocessing utilities
# -------------------------
def fit_preprocessing(df, target_col="Sleep Disorder"):
    df = df.copy()
    # if target_col not present, assume last column is target
    if target_col not in df.columns:
        target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Find categorical vs numeric
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Fill numerics with median
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())

    # Fill categorical NaNs and convert to str
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")

    # encode target if categorical
    target_le = None
    if y.dtype == object or y.dtype.name == "category":
        target_le = LabelEncoder()
        y_enc = target_le.fit_transform(y.astype(str))
    else:
        y_enc = y.values

    # OneHot encode categorical features
    if len(cat_cols) > 0:
        ohe = make_onehotencoder()
        X_cat = ohe.fit_transform(X[cat_cols])
    else:
        ohe = None
        X_cat = np.zeros((len(X), 0))

    # Scale numeric features
    scaler = StandardScaler()
    if len(num_cols) > 0:
        X_num = scaler.fit_transform(X[num_cols].values.astype(float))
    else:
        X_num = np.zeros((len(X), 0))

    X_processed = np.hstack([X_num, X_cat])
    preproc = {"ohe": ohe, "scaler": scaler, "num_cols": num_cols, "cat_cols": cat_cols, "target_le": target_le}
    return X_processed, y_enc, preproc

def transform_features(df_new, preproc):
    df_new = df_new.copy()
    num_cols = preproc["num_cols"]
    cat_cols = preproc["cat_cols"]

    # Ensure missing columns present
    for c in num_cols:
        if c not in df_new.columns:
            df_new[c] = 0
    for c in cat_cols:
        if c not in df_new.columns:
            df_new[c] = "NA"

    # numeric transform
    if len(num_cols) > 0:
        X_num = df_new[num_cols].apply(pd.to_numeric, errors="coerce").fillna(df_new[num_cols].median()).values.astype(float)
        X_num = preproc["scaler"].transform(X_num)
    else:
        X_num = np.zeros((len(df_new), 0))

    # categorical transform
    if len(cat_cols) > 0:
        X_cat = preproc["ohe"].transform(df_new[cat_cols].astype(str))
    else:
        X_cat = np.zeros((len(df_new), 0))

    return np.hstack([X_num, X_cat])

# -------------------------
# Lightweight GA tuner for sklearn MLP
# -------------------------
def ga_tune_mlp(X_train, y_train, X_val, y_val, generations=4, pop_size=6, timeout_seconds=90):
    rng = np.random.RandomState(SEED)
    param_space = {
        "hidden": [(32,), (64,), (64, 32), (128, 64), (128,)],
        "alpha": [1e-4, 1e-3, 1e-2],
        "lr": [1e-3, 3e-3, 1e-2],
        "activation": ["relu", "tanh"]
    }

    def rand_ind():
        return {
            "hidden": tuple(rng.choice(param_space["hidden"])),
            "alpha": float(rng.choice(param_space["alpha"])),
            "lr": float(rng.choice(param_space["lr"])),
            "activation": str(rng.choice(param_space["activation"]))
        }

    def evaluate(ind):
        try:
            model = MLPClassifier(hidden_layer_sizes=ind["hidden"],
                                  alpha=ind["alpha"],
                                  learning_rate_init=ind["lr"],
                                  activation=ind["activation"],
                                  max_iter=150, tol=1e-3, random_state=SEED)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return accuracy_score(y_val, preds), model
        except Exception:
            return 0.0, None

    population = [rand_ind() for _ in range(pop_size)]
    best = (None, 0.0)
    start_time = time.time()

    for gen in range(generations):
        scored = []
        for ind in population:
            score, _ = evaluate(ind)
            scored.append((score, ind))
            if time.time() - start_time > timeout_seconds:
                break
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] > best[1]:
            best = (scored[0][1], scored[0][0])
        survivors = [ind for (_, ind) in scored[: max(2, pop_size // 2)]]
        if not survivors:
            survivors = [rand_ind(), rand_ind()]
        # breed
        new_pop = survivors.copy()
        while len(new_pop) < pop_size:
            a, b = rng.choice(survivors, 2, replace=False)
            child = {}
            for k in a.keys():
                child[k] = a[k] if rng.rand() < 0.5 else b[k]
            # mutate
            if rng.rand() < 0.2:
                key = rng.choice(list(child.keys()))
                child[key] = rand_ind()[key]
            new_pop.append(child)
        population = new_pop
        if time.time() - start_time > timeout_seconds:
            break

    best_params = best[0] if best[0] is not None else rand_ind()
    final = MLPClassifier(hidden_layer_sizes=best_params["hidden"],
                          alpha=best_params["alpha"],
                          learning_rate_init=best_params["lr"],
                          activation=best_params["activation"],
                          max_iter=400, tol=1e-4, random_state=SEED)
    final.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
    return final, best[1], best_params

# -------------------------
# UI Navigation (sidebar)
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Upload Dataset", "Train & Compare", "Manual Predict", "Bulk Predict", "Interpretability"])

# -------------------------
# Home page
# -------------------------
if page == "Home":
    st.title("💤 Sleep Disorder Classification")
    st.subheader("Detecting Sleep Disorders with Machine Learning")
    st.markdown("""
    **Why sleep matters**
    - Sleep is critical for memory consolidation, emotional regulation and metabolic health.
    - Common sleep disorders include insomnia, sleep apnea, restless leg syndrome, and narcolepsy.
    - Automated screening accelerates detection and referral for medical care.

    **How to use this app**
    1. Upload your dataset under **Upload Dataset**. The dataset should include features and a target column named `Sleep Disorder` (or the last column will be treated as target).
    2. Go to **Train & Compare** to train 3 models (ANN, ANN+GA, Hybrid ANN→XGBoost).
    3. Use **Manual Predict** or **Bulk Predict** for new data.
    4. Inspect feature importance in **Interpretability**.
    """)

    if st.button("Go to Upload Dataset"):
        st.experimental_set_query_params(page="Upload Dataset")
        st.sidebar.radio("", ["Home", "Upload Dataset", "Train & Compare", "Manual Predict", "Bulk Predict", "Interpretability"], index=1)

# -------------------------
# Upload Dataset
# -------------------------
elif page == "Upload Dataset":
    st.header("📂 Upload Dataset (single upload per session)")
    uploaded = st.file_uploader("Upload CSV (features + target). Target column should be named 'Sleep Disorder' or will be assumed last column.", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            # if target column missing, rename last column
            if "Sleep Disorder" not in df.columns:
                st.warning("No 'Sleep Disorder' column found — the last column will be used as target and renamed to 'Sleep Disorder'.")
                df = df.rename(columns={df.columns[-1]: "Sleep Disorder"})
            # simple Blood Pressure parsing if present
            if "Blood Pressure" in df.columns:
                try:
                    bp = df["Blood Pressure"].astype(str).str.split("/", expand=True)
                    if bp.shape[1] >= 2:
                        df["Systolic_BP"] = pd.to_numeric(bp[0], errors="coerce")
                        df["Diastolic_BP"] = pd.to_numeric(bp[1], errors="coerce")
                        df.drop(columns=["Blood Pressure"], inplace=True)
                except Exception:
                    pass
            st.session_state.df = df
            st.success("Dataset uploaded and stored in session.")
            st.write("Preview:")
            st.dataframe(df.head(8))
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
    else:
        if st.session_state.df is not None:
            st.info("Dataset already loaded in session.")
            st.dataframe(st.session_state.df.head(5))

# -------------------------
# Train & Compare
# -------------------------
elif page == "Train & Compare":
    st.header("🚀 Train & Compare Models (ANN, ANN+GA, Hybrid ANN→XGBoost)")
    if st.session_state.df is None:
        st.warning("Upload dataset first (Upload Dataset tab).")
    else:
        df = st.session_state.df.copy()
        target_col = "Sleep Disorder"
        if target_col not in df.columns:
            st.error("No 'Sleep Disorder' target column found.")
            st.stop()

        st.info("Fitting preprocessing (OneHot for categoricals, StandardScaler for numerics).")
        X_all, y_all, preproc = fit_preprocessing(df, target_col)
        st.session_state.preproc = preproc

        # Optional balancing
        use_smt = st.checkbox("Apply SMOTETomek balancing (if package installed)", value=False)
        if use_smt and SMOTETomek is not None:
            try:
                smt = SMOTETomek(random_state=SEED)
                X_res, y_res = smt.fit_resample(X_all, y_all)
                st.success("SMOTETomek applied.")
            except Exception:
                X_res, y_res = X_all, y_all
                st.warning("SMOTETomek failed — proceeding without it.")
        else:
            X_res, y_res = X_all, y_all

        # train/val/test split
        strat = y_res if len(np.unique(y_res)) > 1 else None
        X_temp, X_test, y_temp, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=SEED, stratify=strat)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=SEED, stratify=y_temp if len(np.unique(y_temp))>1 else None)

        st.write("Shapes — train:", X_train.shape, "val:", X_val.shape, "test:", X_test.shape)

        if st.button("Run training (fast settings)"):
            t0 = time.time()
            results = {}
            trained = {}

            # Model 1: ANN baseline
            st.write("Training ANN baseline...")
            ann = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", solver="adam", max_iter=400, random_state=SEED)
            ann.fit(X_train, y_train)
            ann_pred = ann.predict(X_test)
            ann_acc = accuracy_score(y_test, ann_pred)
            results["ANN"] = ann_acc
            trained["ANN"] = ann
            st.write(f"ANN accuracy: {ann_acc:.4f}")

            # Model 2: ANN + GA (lightweight)
            st.write("Tuning ANN with lightweight GA...")
            try:
                ga_model, ga_val_best, ga_params = ga_tune_mlp(X_train, y_train, X_val, y_val, generations=4, pop_size=6, timeout_seconds=90)
                ga_pred = ga_model.predict(X_test)
                ga_acc = accuracy_score(y_test, ga_pred)
                results["ANN+GA"] = ga_acc
                trained["ANN+GA"] = ga_model
                st.write(f"ANN+GA accuracy: {ga_acc:.4f} (best val during GA: {ga_val_best:.4f})")
                st.write("GA best params:", ga_params)
            except Exception as e:
                st.warning("GA tuning failed: " + str(e))
                results["ANN+GA"] = 0.0

            # Model 3: Hybrid ANN -> XGBoost stacking (if available)
            if XGBClassifier is not None:
                st.write("Training Hybrid model (ANN -> XGBoost stacking)...")
                try:
                    stack_ann = MLPClassifier(hidden_layer_sizes=(128,64), activation="relu", solver="adam", max_iter=300, random_state=SEED)
                    stack_ann.fit(X_train, y_train)
                    ann_train_proba = stack_ann.predict_proba(X_train)
                    ann_test_proba = stack_ann.predict_proba(X_test)
                    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric="mlogloss", random_state=SEED)
                    xgb.fit(ann_train_proba, y_train)
                    hybrid_pred = xgb.predict(ann_test_proba)
                    hybrid_acc = accuracy_score(y_test, hybrid_pred)
                    results["Hybrid(ANN->XGB)"] = hybrid_acc
                    trained["Hybrid(ANN->XGB)"] = {"ann": stack_ann, "xgb": xgb}
                    st.write(f"Hybrid accuracy: {hybrid_acc:.4f}")
                except Exception as e:
                    st.warning("Hybrid training failed: " + str(e))
                    results["Hybrid(ANN->XGB)"] = 0.0
            else:
                st.info("xgboost not installed — Hybrid model skipped.")
                results["Hybrid(ANN->XGB)"] = 0.0

            # Show cards
            st.markdown("### Model accuracies")
            cols = st.columns(len(results))
            palette = ["#dff7ec", "#eef7ff", "#fff9e6", "#fff0f0"]
            i = 0
            for name, acc in results.items():
                with cols[i]:
                    st.markdown(f"<div class='card'><div class='label'>{name}</div><div class='metric'>{acc*100:.2f}%</div><div class='label'>Accuracy</div></div>", unsafe_allow_html=True)
                i += 1

            # Save to session and disk (best only)
            st.session_state.models = trained
            best_name = max(results.items(), key=lambda x: x[1])[0]
            best_score = results[best_name]
            st.session_state.best["name"] = best_name
            st.session_state.best["score"] = best_score

            save_obj = {"preproc": preproc, "best_name": best_name}
            if best_name == "Hybrid(ANN->XGB)":
                save_obj["model_info"] = {"type": "hybrid", "ann": trained[best_name]["ann"], "xgb": trained[best_name]["xgb"]}
            else:
                save_obj["model_info"] = {"type": best_name, "model": trained[best_name]}

            saved_flag = save_to_disk(save_obj, MODEL_FILE)
            st.session_state.saved = saved_flag
            st.success(f"Training finished in {time.time()-t0:.1f}s. Best: {best_name} ({best_score*100:.2f}%). Saved: {saved_flag}")

            # Confusion for best
            try:
                st.markdown("Confusion matrix for best model")
                if best_name == "Hybrid(ANN->XGB)":
                    ann_m = trained[best_name]["ann"]
                    xgb_m = trained[best_name]["xgb"]
                    preds = xgb_m.predict(ann_m.predict_proba(X_test))
                else:
                    preds = trained[best_name].predict(X_test)
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            except Exception as e:
                st.warning("Could not plot confusion matrix: " + str(e))

# -------------------------
# Manual Predict
# -------------------------
elif page == "Manual Predict":
    st.header("🔮 Manual Prediction")
    if st.session_state.df is None or st.session_state.preproc is None or st.session_state.models == {}:
        st.warning("Please upload data and train models first.")
    else:
        raw = st.session_state.df.copy()
        feat_cols = [c for c in raw.columns if c != "Sleep Disorder"]
        st.markdown("Enter feature values (text for categorical, numeric for numeric).")
        vals = {}
        cols = st.columns(2)
        for i, c in enumerate(feat_cols):
            if raw[c].dtype == object:
                vals[c] = cols[i%2].text_input(c, value=str(raw[c].mode().iloc[0]) if not raw[c].mode().empty else "NA")
            else:
                med = float(raw[c].median()) if not np.isnan(raw[c].median()) else 0.0
                vals[c] = cols[i%2].number_input(c, value=med)

        if st.button("Predict (using best model)"):
            try:
                sample_df = pd.DataFrame([vals])
                Xs = transform_features(sample_df, st.session_state.preproc)
                best_name = st.session_state.best["name"]
                if best_name is None:
                    st.error("No trained best model found. Train models first.")
                else:
                    if best_name == "Hybrid(ANN->XGB)":
                        ann_m = st.session_state.models[best_name]["ann"]
                        xgb_m = st.session_state.models[best_name]["xgb"]
                        pred = xgb_m.predict(ann_m.predict_proba(Xs))[0]
                    else:
                        pred = st.session_state.models[best_name].predict(Xs)[0]
                    t_le = st.session_state.preproc.get("target_le", None)
                    if t_le is not None:
                        pred_label = t_le.inverse_transform([int(pred)])[0]
                    else:
                        pred_label = pred
                    st.success(f"Predicted Sleep Disorder: {pred_label}")
            except Exception as e:
                st.error("Prediction failed: " + str(e))

# -------------------------
# Bulk Predict
# -------------------------
elif page == "Bulk Predict":
    st.header("📥 Bulk Prediction (CSV)")
    if st.session_state.df is None or st.session_state.preproc is None or st.session_state.models == {}:
        st.warning("Upload data and train models first.")
    else:
        upload = st.file_uploader("Upload CSV for prediction (features only)", type=["csv"])
        if upload:
            try:
                newdf = pd.read_csv(upload)
                if "Sleep Disorder" in newdf.columns:
                    newdf = newdf.drop(columns=["Sleep Disorder"])
                Xnew = transform_features(newdf, st.session_state.preproc)
                best_name = st.session_state.best["name"]
                if best_name == "Hybrid(ANN->XGB)":
                    ann_m = st.session_state.models[best_name]["ann"]
                    xgb_m = st.session_state.models[best_name]["xgb"]
                    preds = xgb_m.predict(ann_m.predict_proba(Xnew))
                else:
                    preds = st.session_state.models[best_name].predict(Xnew)
                t_le = st.session_state.preproc.get("target_le", None)
                if t_le is not None:
                    pred_labels = t_le.inverse_transform(preds.astype(int))
                else:
                    pred_labels = preds
                out = newdf.copy()
                out["Predicted_Sleep_Disorder"] = pred_labels
                st.dataframe(out.head(50))
                st.download_button("Download predictions CSV", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
            except Exception as e:
                st.error("Bulk prediction failed: " + str(e))

# -------------------------
# Interpretability
# -------------------------
elif page == "Interpretability":
    st.header("🔍 Interpretability — Permutation Feature Importance")
    if st.session_state.df is None or st.session_state.preproc is None or st.session_state.models == {}:
        st.warning("Upload and train first.")
    else:
        df = st.session_state.df.copy()
        features = [c for c in df.columns if c != "Sleep Disorder"]
        try:
            X_all = transform_features(df[features], st.session_state.preproc)
            y_all = df["Sleep Disorder"]
            if st.session_state.preproc.get("target_le", None) is not None:
                y_all_enc = st.session_state.preproc["target_le"].transform(y_all.astype(str))
            else:
                y_all_enc = y_all.values

            st.info("Computing permutation importance (may take some time)...")
            best_name = st.session_state.best["name"]
            if best_name is None:
                st.error("No best model found.")
            else:
                # Build a predict function for permutation_importance use
                if best_name == "Hybrid(ANN->XGB)":
                    # Use the xgb estimator (it expects ANN-proba features)
                    estimator = st.session_state.models[best_name]["xgb"]
                    # We will evaluate importance on the processed X by mapping X->ann proba inside permutation scoring
                    # But sklearn's permutation_importance expects estimator that takes X directly; we can use a wrapper
                    class HybridWrapper:
                        def __init__(self, ann, xgb):
                            self.ann = ann
                            self.xgb = xgb
                        def predict(self, Xp):
                            return self.xgb.predict(self.ann.predict_proba(Xp))
                    wrapper = HybridWrapper(st.session_state.models[best_name]["ann"], st.session_state.models[best_name]["xgb"])
                    estimator_for_perm = wrapper
                else:
                    estimator_for_perm = st.session_state.models[best_name]

                res = permutation_importance(estimator_for_perm, X_all, y_all_enc, n_repeats=8, random_state=SEED, n_jobs=1)
                importances = res.importances_mean

                # Map processed columns back to original features (approx) — group OHE columns
                num_cols = st.session_state.preproc["num_cols"]
                cat_cols = st.session_state.preproc["cat_cols"]
                proc_names = []
                if len(num_cols) > 0:
                    proc_names.extend(num_cols)
                if len(cat_cols) > 0:
                    ohe = st.session_state.preproc["ohe"]
                    for i, col in enumerate(cat_cols):
                        cats = ohe.categories_[i]
                        proc_names.extend([f"{col}__{c}" for c in cats])
                # Build df
                proc_df = pd.DataFrame({"proc_col": proc_names, "importance": importances})
                proc_df["orig"] = proc_df["proc_col"].apply(lambda p: p.split("__")[0] if "__" in p else p)
                agg = proc_df.groupby("orig")["importance"].sum().reset_index().sort_values("importance", ascending=False)
                st.dataframe(agg)
                fig, ax = plt.subplots(figsize=(8, max(4, 0.4*len(agg))))
                sns.barplot(x="importance", y="orig", data=agg, ax=ax)
                ax.set_xlabel("Mean permutation importance")
                ax.set_ylabel("Feature")
                st.pyplot(fig)
        except Exception as e:
            st.error("Interpretability failed: " + str(e))

# End of app
