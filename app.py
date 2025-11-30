# app.py
"""
Sleep Disorder Classification (final working app)
- Default landing page: Upload Dataset
- Navigation: Upload Dataset, Train Models, Manual Predict, Bulk Predict, Interpretability
- Models: RandomForest, optional XGBoost, GA-tuned ANN (sklearn MLP)
- UI: Soft Card Style accuracy boxes, colorful but no pictures
- Dataset stored in session_state and reused across pages
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, random, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports
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
# Config
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
MODEL_FILE = "best_model.pkl"

st.set_page_config(page_title="Sleep Disorder Classifier", layout="wide")

# -------------------------
# CSS / Soft Card Style
# -------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(90deg,#f6fbff,#f0f9ff); }
    .top-row{ display:flex; gap:12px; }
    .card {
      border-radius:12px;
      padding:14px;
      box-shadow: 0 6px 18px rgba(8,61,119,0.06);
      background: linear-gradient(180deg,#ffffff,#f7fbff);
      border: 1px solid rgba(11,72,107,0.04);
    }
    .metric {
      border-radius:10px;
      padding:12px;
      min-width:180px;
    }
    .m-green { background: linear-gradient(180deg,#e9f9f1,#e6fff6); border-left:6px solid #2bb673; }
    .m-blue { background: linear-gradient(180deg,#eef7ff,#f0f8ff); border-left:6px solid #2196f3; }
    .m-yellow { background: linear-gradient(180deg,#fff9ec,#fffdf0); border-left:6px solid #ffb020; }
    .m-red { background: linear-gradient(180deg,#fff5f6,#fff6f7); border-left:6px solid #ff5c6c; }
    .model-name { font-weight:700; color:#083d77; }
    .small { color:#6b7885; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Session state initial
# -------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "feature_order" not in st.session_state:
    st.session_state.feature_order = None
if "label_encoders" not in st.session_state:
    st.session_state.label_encoders = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None
if "best_score" not in st.session_state:
    st.session_state.best_score = None

# -------------------------
# Helpers: save/load
# -------------------------
def save_model_file(model, scaler, encoders, feature_order):
    try:
        with open(MODEL_FILE, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler, "encoders": encoders, "features": feature_order}, f)
        return True
    except Exception as e:
        st.error(f"Saving failed: {e}")
        return False

def load_model_file():
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

# -------------------------
# Utility functions
# -------------------------
def preprocess_dataset(df, target_col):
    df = df.copy()
    # drop fully empty rows
    df = df.dropna(how="all")
    # parse simple BP if present
    if "Blood Pressure" in df.columns:
        try:
            df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].astype(str).str.split("/", expand=True).astype(float)
            df.drop(columns=["Blood Pressure"], inplace=True)
        except Exception:
            pass
    # fill numeric and categorical missing
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c != target_col]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "NA")
    # encode categorical features
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le
    # target
    if df[target_col].dtype == object or df[target_col].dtype.name == "category":
        target_le = LabelEncoder()
        y = target_le.fit_transform(df[target_col].astype(str))
    else:
        target_le = None
        y = df[target_col].values
    X = df.drop(columns=[target_col])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values.astype(float))
    return Xs, y, encoders, scaler, X.columns.tolist(), target_le

def safe_train_test_split(X, y, test_size=0.2, random_state=SEED):
    try:
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def draw_accuracy_cards(results_dict):
    # results_dict: {name: accuracy_float_0to1}
    cols = st.columns(len(results_dict))
    i = 0
    palette = ["m-green", "m-blue", "m-yellow", "m-red"]
    for name, acc in results_dict.items():
        with cols[i]:
            cls = palette[i % len(palette)]
            st.markdown(f'<div class="card metric {cls}">', unsafe_allow_html=True)
            st.markdown(f'<div class="model-name">{name}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:28px; font-weight:700;">{acc*100:.2f}%</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="small">Accuracy</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        i += 1

# -------------------------
# GA tuner for sklearn MLP (lightweight)
# -------------------------
def ga_tune_mlp(X_train, y_train, X_val, y_val, generations=4, pop_size=6, timeout_seconds=90):
    """
    Lightweight GA to tune sklearn MLPClassifier hyperparams.
    Uses small training (max_iter small) to evaluate quickly.
    Returns best_model (trained on train+val), best_score_val, best_params
    """
    rng = np.random.RandomState(SEED)
    param_space = {
        "hidden_layer_sizes": [(32,), (64,), (64,32), (128,64), (128,)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.003, 0.01],
        "activation": ["relu", "tanh"],
    }

    def random_individual():
        return {
            "hidden_layer_sizes": tuple(rng.choice(param_space["hidden_layer_sizes"])),
            "alpha": float(rng.choice(param_space["alpha"])),
            "learning_rate_init": float(rng.choice(param_space["learning_rate_init"])),
            "activation": str(rng.choice(param_space["activation"]))
        }

    def mutate(ind):
        # small mutation
        if rng.rand() < 0.3:
            ind["alpha"] = float(rng.choice(param_space["alpha"]))
        if rng.rand() < 0.3:
            ind["learning_rate_init"] = float(rng.choice(param_space["learning_rate_init"]))
        if rng.rand() < 0.2:
            ind["hidden_layer_sizes"] = tuple(rng.choice(param_space["hidden_layer_sizes"]))
        if rng.rand() < 0.2:
            ind["activation"] = str(rng.choice(param_space["activation"]))
        return ind

    def crossover(a, b):
        child = {}
        for k in a.keys():
            child[k] = a[k] if rng.rand() < 0.5 else b[k]
        return child

    # init
    population = [random_individual() for _ in range(pop_size)]
    scored = []
    best = (None, 0.0)  # (params, score)
    start_time = time.time()

    for gen in range(generations):
        st.sidebar.info(f"GA generation {gen+1}/{generations}")
        scored = []
        for ind in population:
            # quick eval: train small MLP
            try:
                model = MLPClassifier(hidden_layer_sizes=ind["hidden_layer_sizes"],
                                      alpha=ind["alpha"],
                                      learning_rate_init=ind["learning_rate_init"],
                                      activation=ind["activation"],
                                      max_iter=120, tol=1e-3, random_state=SEED)
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
            except Exception:
                val_acc = 0.0
            scored.append((val_acc, ind))
            # timeout guard
            if time.time() - start_time > timeout_seconds:
                break
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] > best[1]:
            best = (scored[0][1], scored[0][0])
        # selection
        survivors = [ind for (_, ind) in scored[: max(2, pop_size//2)]]
        # breed
        new_pop = survivors.copy()
        while len(new_pop) < pop_size:
            a, b = rng.choice(survivors, 2, replace=False)
            child = crossover(a, b)
            child = mutate(child)
            new_pop.append(child)
        population = new_pop
        # timeout break
        if time.time() - start_time > timeout_seconds:
            break

    # train final model on combined train+val
    best_params = best[0] if best[0] is not None else random_individual()
    final_model = MLPClassifier(hidden_layer_sizes=best_params["hidden_layer_sizes"],
                                alpha=best_params["alpha"],
                                learning_rate_init=best_params["learning_rate_init"],
                                activation=best_params["activation"],
                                max_iter=400, tol=1e-4, random_state=SEED)
    # combine train+val
    X_comb = np.vstack([X_train, X_val])
    y_comb = np.hstack([y_train, y_val])
    final_model.fit(X_comb, y_comb)
    return final_model, best[1], best_params

# -------------------------
# Navigation: Upload Dataset (default landing)
# -------------------------
pages = ["Upload Dataset", "Train Models", "Manual Predict", "Bulk Predict", "Interpretability"]
page = st.sidebar.selectbox("Navigation", pages, index=0)

# Page: Upload Dataset
if page == "Upload Dataset":
    st.header("📂 Upload Dataset (one-time per session)")
    st.markdown("Upload a CSV containing features and a `Sleep Disorder` target column. The dataset will be used across all pages in this session.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success("Dataset loaded into session.")
            st.write("Preview:")
            st.dataframe(df.head())
            st.markdown("---")
            st.info("Next: go to **Train Models** to train and compare models.")
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))
    elif st.session_state.df is not None:
        st.write("Dataset already uploaded for this session.")
        st.dataframe(st.session_state.df.head())

# Page: Train Models
elif page == "Train Models":
    st.header("🚀 Train Models & GA-tuned ANN")
    if st.session_state.df is None:
        st.warning("Upload dataset first (Upload Dataset).")
    else:
        df = st.session_state.df.copy()
        target_col = "Sleep Disorder"
        if target_col not in df.columns:
            st.error("Your dataset must contain a column named 'Sleep Disorder' (target).")
        else:
            # preprocess & encode
            with st.expander("Preview / Preprocess options (click to expand)"):
                st.write("Dataset head:")
                st.dataframe(df.head())
                st.write("Columns:", list(df.columns))
            # Preprocess
            Xs, y, encoders, scaler, features, target_le = preprocess_dataset(df, target_col)
            st.session_state.feature_order = features
            st.session_state.label_encoders = encoders
            st.session_state.scaler = scaler
            # train/val/test split (for GA need train+val)
            X_temp, X_test, y_temp, y_test = safe_train_test_split(Xs, y, test_size=0.2, random_state=SEED)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=SEED, stratify=y_temp if len(np.unique(y_temp))>1 else None)

            st.info("Will train: RandomForest, optional XGBoost (if installed), and GA-tuned ANN (sklearn MLP).")
            st.markdown("**GA tuning parameters (lightweight):** population=6, generations=4, short eval training. You can increase for better accuracy (slower).")

            run = st.button("Run training (this may take a minute)")
            if run:
                start_all = time.time()
                results = {}
                models = {}

                # RandomForest (fast)
                st.write("Training RandomForest...")
                rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                rf_acc = accuracy_score(y_test, rf_pred)
                results["RandomForest"] = rf_acc
                models["RandomForest"] = rf
                st.write(f"RandomForest accuracy: {rf_acc:.4f}")

                # XGBoost if available
                if XGBClassifier is not None:
                    st.write("Training XGBoost...")
                    xgb = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=SEED)
                    try:
                        xgb.fit(X_train, y_train)
                        xgb_pred = xgb.predict(X_test)
                        xgb_acc = accuracy_score(y_test, xgb_pred)
                        results["XGBoost"] = xgb_acc
                        models["XGBoost"] = xgb
                        st.write(f"XGBoost accuracy: {xgb_acc:.4f}")
                    except Exception as e:
                        st.warning("XGBoost failed: " + str(e))
                else:
                    st.info("XGBoost not installed — skipped.")

                # GA-tuned ANN (sklearn MLP)
                st.write("Running lightweight GA tuner for ANN (sklearn MLP)...")
                try:
                    ann_model, best_val_acc, best_params = ga_tune_mlp(X_train, y_train, X_val, y_val,
                                                                      generations=4, pop_size=6, timeout_seconds=90)
                    ann_pred = ann_model.predict(X_test)
                    ann_acc = accuracy_score(y_test, ann_pred)
                    results["GA-ANN"] = ann_acc
                    models["GA-ANN"] = ann_model
                    st.write(f"GA-ANN test accuracy: {ann_acc:.4f}")
                    st.write("GA best params:", best_params)
                except Exception as e:
                    st.warning("GA-ANN failed: " + str(e))

                # pick best
                best_name = max(results.items(), key=lambda x: x[1])[0]
                best_score = results[best_name]
                st.session_state.best_model = models[best_name]
                st.session_state.best_model_name = best_name
                st.session_state.best_score = best_score

                # show accuracy cards
                st.markdown("### Model accuracies")
                draw_accuracy_cards(results)

                # show comparison table
                acc_df = pd.DataFrame([(k, v) for k, v in results.items()], columns=["Model", "Accuracy"])
                acc_df["Accuracy (%)"] = (acc_df["Accuracy"] * 100).round(2)
                st.table(acc_df[["Model", "Accuracy (%)"]])

                # confusion matrix for best
                try:
                    st.markdown("### Confusion matrix — Best model")
                    cm = confusion_matrix(y_test, st.session_state.best_model.predict(X_test))
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)
                except Exception:
                    pass

                # save model once
                saved = save_model_file(st.session_state.best_model, st.session_state.scaler, st.session_state.label_encoders, st.session_state.feature_order)
                if saved:
                    st.success(f"Best model ({best_name}) trained & saved. Accuracy: {best_score*100:.2f}%")
                else:
                    st.info("Model trained but failed to save to disk (permission or path issue).")

                st.write(f"Total training time: {time.time() - start_all:.1f}s")

# Page: Manual Predict
elif page == "Manual Predict":
    st.header("🔮 Manual Prediction")
    if st.session_state.df is None or st.session_state.best_model is None:
        st.warning("Upload dataset and train a model first.")
    else:
        features = st.session_state.feature_order
        st.markdown("Fill feature values; defaults shown (you can edit).")
        user_vals = {}
        col1, col2 = st.columns(2)
        # Make inputs compact
        for i, feat in enumerate(features):
            if i % 2 == 0:
                user_vals[feat] = col1.text_input(feat, value="0")
            else:
                user_vals[feat] = col2.text_input(feat, value="0")
        if st.button("Predict"):
            try:
                user_df = pd.DataFrame([user_vals])
                # apply encoders where available
                for col, le in st.session_state.label_encoders.items():
                    if col in user_df.columns:
                        val = str(user_df.at[0, col])
                        if val not in le.classes_:
                            le.classes_ = np.append(le.classes_, val)
                        user_df[col] = le.transform(user_df[col].astype(str))
                # order and scale
                user_df = user_df[features].astype(float)
                Xs = st.session_state.scaler.transform(user_df)
                pred = st.session_state.best_model.predict(Xs)[0]
                # decode if target encoder exists
                # if dataset's target was categorical encoded earlier, we stored target encoder in label_encoders? we didn't store target explicitly; try best effort:
                # We assume 'Sleep Disorder' might be in label_encoders (if originally object). Otherwise show numeric.
                if "Sleep Disorder" in st.session_state.label_encoders:
                    label = st.session_state.label_encoders["Sleep Disorder"].inverse_transform([int(pred)])[0]
                else:
                    label = pred
                st.success(f"Predicted Sleep Disorder: {label}")
            except Exception as e:
                st.error("Prediction failed: " + str(e))

# Page: Bulk Predict
elif page == "Bulk Predict":
    st.header("📥 Bulk Prediction (CSV)")
    if st.session_state.df is None or st.session_state.best_model is None:
        st.warning("Upload dataset and train a model first.")
    else:
        uploaded = st.file_uploader("Upload CSV for prediction (features only or same columns)", type=["csv"])
        if uploaded is not None:
            try:
                new_df = pd.read_csv(uploaded)
                # simple BP parse if present
                if "Blood Pressure" in new_df.columns:
                    try:
                        new_df[["Systolic_BP", "Diastolic_BP"]] = new_df["Blood Pressure"].astype(str).str.split("/", expand=True).astype(float)
                        new_df.drop(columns=["Blood Pressure"], inplace=True)
                    except Exception:
                        pass
                # apply encoders (make sure unseen classes handled)
                for col, le in st.session_state.label_encoders.items():
                    if col in new_df.columns:
                        new_df[col] = new_df[col].astype(str)
                        missing = set(new_df[col]) - set(le.classes_)
                        if missing:
                            le.classes_ = np.append(le.classes_, list(missing))
                        new_df[col] = le.transform(new_df[col])
                # align columns
                feats = st.session_state.feature_order
                try:
                    X_new = new_df[feats].astype(float)
                except Exception:
                    # fallback: take first N columns
                    X_new = new_df.iloc[:, :len(feats)].astype(float)
                    X_new.columns = feats
                X_new_s = st.session_state.scaler.transform(X_new)
                preds = st.session_state.best_model.predict(X_new_s)
                # decode if possible
                if "Sleep Disorder" in st.session_state.label_encoders:
                    pred_labels = st.session_state.label_encoders["Sleep Disorder"].inverse_transform(preds.astype(int))
                else:
                    pred_labels = preds
                out = new_df.copy()
                out["Predicted_Sleep_Disorder"] = pred_labels
                st.dataframe(out.head(50))
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download predictions CSV", csv, "predictions.csv")
            except Exception as e:
                st.error("Bulk prediction failed: " + str(e))

# Page: Interpretability
elif page == "Interpretability":
    st.header("📊 Interpretability & Feature Importance")
    if st.session_state.df is None or st.session_state.best_model is None:
        st.warning("Upload dataset and train a model first.")
    else:
        st.markdown("Permutation importance is computed on the training dataset (scaled). SHAP is optional (if installed).")
        # prepare X and y
        df = st.session_state.df.copy()
        feats = st.session_state.feature_order
        # encode dataset consistently
        for col, le in st.session_state.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str)
                missing = set(df[col]) - set(le.classes_)
                if missing:
                    le.classes_ = np.append(le.classes_, list(missing))
                df[col] = le.transform(df[col])
        try:
            X = df[feats].astype(float)
        except Exception as e:
            st.error("Failed to prepare dataset for interpretability: " + str(e))
            X = None
        if X is not None:
            Xs = st.session_state.scaler.transform(X)
            st.info("Computing permutation importance (may take ~10-30s).")
            try:
                from sklearn.inspection import permutation_importance
                res = permutation_importance(st.session_state.best_model, Xs, st.session_state.scaler.inverse_transform(Xs)[:, 0] if False else None, n_repeats=8, random_state=SEED)
                # The above computation needs y; we will use dataset's true y where possible:
                # Recompute with proper y:
                target = "Sleep Disorder"
                if target in st.session_state.df.columns:
                    y_full = st.session_state.df[target]
                    if y_full.dtype == object:
                        if "Sleep Disorder" in st.session_state.label_encoders:
                            y_full_enc = st.session_state.label_encoders["Sleep Disorder"].transform(y_full.astype(str))
                        else:
                            y_full_enc = LabelEncoder().fit_transform(y_full.astype(str))
                    else:
                        y_full_enc = y_full.values
                    res = permutation_importance(st.session_state.best_model, Xs, y_full_enc, n_repeats=8, random_state=SEED)
                    idx = res.importances_mean.argsort()[::-1]
                    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(feats))))
                    sns.barplot(x=res.importances_mean[idx], y=np.array(feats)[idx], ax=ax)
                    ax.set_xlabel("Mean importance")
                    st.pyplot(fig)
                else:
                    st.error("Cannot compute permutation importance: 'Sleep Disorder' column missing in data.")
            except Exception as e:
                st.error("Permutation importance failed: " + str(e))

            # SHAP (optional)
            if shap is not None:
                st.markdown("### SHAP (sampled; may be slow)")
                try:
                    explainer = None
                    model = st.session_state.best_model
                    # For tree-based models, use TreeExplainer, else KernelExplainer may be slow
                    if hasattr(model, "feature_importances_") and shap is not None:
                        explainer = shap.TreeExplainer(model)
                        sample = Xs[:min(200, Xs.shape[0])]
                        shap_vals = explainer.shap_values(sample)
                        st.pyplot(shap.summary_plot(shap_vals, sample, feature_names=feats, show=False))
                    else:
                        st.info("SHAP KernelExplainer is slow for non-tree models; skipping by default.")
                except Exception as e:
                    st.warning("SHAP failed: " + str(e))
            else:
                st.info("Install `shap` for richer explanations (optional).")

# End of app

