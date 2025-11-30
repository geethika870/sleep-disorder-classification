# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, time, random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports (handled gracefully)
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from imblearn.combine import SMOTETomek
except Exception:
    SMOTETomek = None

# -----------------------
# Settings
# -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
MODEL_FILE = "best_artifacts.pkl"

st.set_page_config(page_title="Sleep Disorder Classifier", layout="wide")
sns.set_style("whitegrid")

# -----------------------
# CSS / UI
# -----------------------
st.markdown("""
<style>
body { background-color: #f7fbff; }
.header { font-size: 34px; color: #0b3d91; font-weight:700; }
.card { border-radius:10px; padding:12px; background: #ffffff; border:1px solid #e6eefb; }
.small { color:#6c757d; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Helpers: OneHotEncoder compatibility
# -----------------------
def make_onehotencoder():
    from sklearn.preprocessing import OneHotEncoder
    try:
        # scikit-learn >= 1.2 uses sparse_output
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # older versions use sparse
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# -----------------------
# Safe train_test_split helper
# -----------------------
def safe_train_test_split(X, y, test_size=0.25, random_state=SEED):
    # y expected as pandas Series or 1d array
    try:
        y_series = pd.Series(y)
        unique_counts = y_series.value_counts()
        if len(unique_counts) < 2:
            st.warning("⚠ Target has only one class — stratified split is disabled.")
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        if any(unique_counts < 3):
            st.warning("⚠ Some classes have < 3 samples — stratified split disabled to avoid error.")
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    except Exception:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

# -----------------------
# Preprocessing: build and transform
# -----------------------
def build_preprocessor(df, target_col="Sleep Disorder"):
    # drop target if present in features
    cols = df.columns.tolist()
    if target_col not in cols:
        target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    # numeric and categorical detection
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    # remove target if somehow present
    if target_col in num_cols:
        num_cols.remove(target_col)
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", make_onehotencoder())])

    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", num_pipe, num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", cat_pipe, cat_cols))

    if len(transformers) == 0:
        # no features? create trivial passthrough
        preproc = ColumnTransformer([("none", "passthrough", X.columns.tolist())])
    else:
        preproc = ColumnTransformer(transformers, remainder="drop")
    return preproc, num_cols, cat_cols

def fit_preprocessor(df, target_col="Sleep Disorder"):
    preproc, num_cols, cat_cols = build_preprocessor(df, target_col)
    X = df.drop(columns=[target_col])
    # ensure numeric columns are numeric
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")
    preproc.fit(X)
    return preproc, num_cols, cat_cols

def transform_with_preproc(df_new, preproc, num_cols, cat_cols):
    # ensure columns exist
    df_new = df_new.copy()
    for c in num_cols:
        if c not in df_new.columns:
            df_new[c] = 0
    for c in cat_cols:
        if c not in df_new.columns:
            df_new[c] = "NA"
    # coerce numeric columns
    for c in num_cols:
        df_new[c] = pd.to_numeric(df_new[c], errors="coerce").fillna(df_new[c].median())
    for c in cat_cols:
        df_new[c] = df_new[c].astype(str).fillna("NA")
    Xp = preproc.transform(df_new[num_cols + cat_cols if len(cat_cols)>0 else num_cols])
    return Xp

# -----------------------
# GA tuner (lightweight) — tunes feature selection (fast)
# -----------------------
def ga_feature_selector(X_train, y_train, generations=4, pop_size=8, timeout_seconds=60):
    """
    Lightweight GA to produce a boolean mask of selected features.
    This is fast and simple: populations are random masks; we keep best masks by accuracy on a small CV split.
    Returns an index list of selected features.
    """
    rng = np.random.RandomState(SEED)
    n_features = X_train.shape[1]
    # train/val split (2-way)
    try:
        Xt, Xv, yt, yv = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train if len(np.unique(y_train))>1 else None)
    except Exception:
        Xt, Xv, yt, yv = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED)

    def random_mask():
        # ensure at least 2 features
        mask = rng.choice([0,1], size=n_features, p=[0.45,0.55])
        if mask.sum() < 2:
            idx = rng.choice(n_features, 2, replace=False)
            mask[idx] = 1
        return mask

    def score_mask(mask):
        try:
            sel = np.where(mask == 1)[0]
            if len(sel) == 0:
                return 0.0
            clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=250, random_state=SEED)
            clf.fit(Xt[:, sel], yt)
            preds = clf.predict(Xv[:, sel])
            return accuracy_score(yv, preds)
        except Exception:
            return 0.0

    # initial population
    population = [random_mask() for _ in range(pop_size)]
    best_mask = None
    best_score = -1.0
    t0 = time.time()
    for gen in range(generations):
        scored = []
        for mask in population:
            s = score_mask(mask)
            scored.append((s, mask))
            if time.time() - t0 > timeout_seconds:
                break
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] > best_score:
            best_score, best_mask = scored[0][0], scored[0][1]
        # selection: keep top half
        keep = [m for (_, m) in scored[: max(2, pop_size//2)]]
        # breed
        new_pop = keep.copy()
        while len(new_pop) < pop_size:
            a, b = rng.choice(keep, 2, replace=False)
            # crossover
            cross_pt = rng.randint(1, n_features-1)
            child = np.concatenate([a[:cross_pt], b[cross_pt:]])
            # mutation
            if rng.rand() < 0.2:
                i = rng.randint(0, n_features)
                child[i] = 1 - child[i]
            # ensure at least 2 features
            if child.sum() < 2:
                idx = rng.choice(n_features, 2, replace=False)
                child[idx] = 1
            new_pop.append(child)
        population = new_pop
        if time.time() - t0 > timeout_seconds:
            break
    if best_mask is None:
        best_mask = random_mask()
    sel_idx = np.where(best_mask == 1)[0]
    return sel_idx.tolist()

# -----------------------
# Train three models
# -----------------------
def train_ann_model(X_train, y_train, X_test, y_test):
    ann = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=400, random_state=SEED)
    ann.fit(X_train, y_train)
    preds = ann.predict(X_test)
    return accuracy_score(y_test, preds), ann

def train_ann_ga_model(X_train, y_train, X_test, y_test):
    # feature selection via GA
    sel_idx = ga_feature_selector(X_train, y_train, generations=4, pop_size=8, timeout_seconds=60)
    if len(sel_idx) == 0:
        sel_idx = list(range(min(2, X_train.shape[1])))
    Xtr = X_train[:, sel_idx]
    Xte = X_test[:, sel_idx]
    ann = MLPClassifier(hidden_layer_sizes=(96,48), max_iter=500, random_state=SEED)
    ann.fit(Xtr, y_train)
    preds = ann.predict(Xte)
    return accuracy_score(y_test, preds), ann, sel_idx

def train_hybrid_model(X_train, y_train, X_test, y_test):
    if XGBClassifier is None:
        raise RuntimeError("xgboost not installed. Hybrid model unavailable.")
    # Stage 1: ANN to produce probs
    ann = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=SEED)
    ann.fit(X_train, y_train)
    train_proba = ann.predict_proba(X_train)
    test_proba = ann.predict_proba(X_test)
    # Stage 2: XGBoost on ANN probabilities
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric="mlogloss", random_state=SEED)
    xgb.fit(train_proba, y_train)
    preds = xgb.predict(test_proba)
    return accuracy_score(y_test, preds), ann, xgb

# -----------------------
# Save / Load artifacts
# -----------------------
def save_artifacts(path=MODEL_FILE, artifacts=None):
    try:
        with open(path, "wb") as f:
            pickle.dump(artifacts, f)
        return True
    except Exception as e:
        st.error(f"Failed to save artifacts: {e}")
        return False

def load_artifacts(path=MODEL_FILE):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Failed to load artifacts: {e}")
            return None
    return None

# -----------------------
# UI & Navigation (single page with sections)
# -----------------------
st.markdown('<div class="header">Sleep Disorder Classification (single-file app)</div>', unsafe_allow_html=True)
st.write("A streamlined app that trains three models (ANN, ANN+GA, Hybrid ANN→XGBoost), compares them, allows predictions and shows feature importance. Target column name must be **Sleep Disorder**.")

st.markdown("---")

# Section: Upload dataset
st.markdown("### 1) Upload dataset")
uploaded = st.file_uploader("Upload CSV (features + target). Target must be named exactly: 'Sleep Disorder'", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df = None
    if df is not None:
        # if Blood Pressure present, split convenience
        if "Blood Pressure" in df.columns:
            try:
                bp = df["Blood Pressure"].astype(str).str.split("/", expand=True)
                if bp.shape[1] >= 2:
                    df["Systolic_BP"] = pd.to_numeric(bp[0], errors="coerce")
                    df["Diastolic_BP"] = pd.to_numeric(bp[1], errors="coerce")
                    df.drop(columns=["Blood Pressure"], inplace=True)
            except Exception:
                pass
        st.session_state["df"] = df
        st.success("Dataset uploaded into session.")
        st.dataframe(df.head(6))

# load df from session if present
df = st.session_state.get("df", None)

st.markdown("---")

# Section: Train & compare
st.markdown("### 2) Train & Compare Models")
if df is None:
    st.info("Upload a dataset above to enable training.")
else:
    # check target presence
    if "Sleep Disorder" not in df.columns:
        st.error("Target column 'Sleep Disorder' not found. Make sure your CSV has this exact column name.")
    else:
        col_count = df.shape[1] - 1
        st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns (features: {col_count})")

        # Fit preprocessor
        if st.button("Fit Preprocessor & Preview"):
            with st.spinner("Fitting preprocessor..."):
                preproc, num_cols, cat_cols = fit_preprocessor(df, "Sleep Disorder")
                st.session_state["preproc"] = preproc
                st.session_state["num_cols"] = num_cols
                st.session_state["cat_cols"] = cat_cols
                st.success("Preprocessor fitted and stored in session.")
                st.write("Numeric cols:", num_cols)
                st.write("Categorical cols:", cat_cols)

        # Train models button
        if st.button("Train the 3 models (ANN, ANN+GA, Hybrid)"):
            if "preproc" not in st.session_state:
                st.warning("Fit the preprocessor first (click 'Fit Preprocessor & Preview').")
            else:
                preproc = st.session_state["preproc"]
                num_cols = st.session_state.get("num_cols", [])
                cat_cols = st.session_state.get("cat_cols", [])

                # Prepare X,y
                X_raw = df.drop(columns=["Sleep Disorder"])
                y_raw = df["Sleep Disorder"]
                # label-encode target
                target_le = LabelEncoder()
                y_enc = target_le.fit_transform(y_raw.astype(str))
                st.session_state["target_le"] = target_le

                # transform features
                # ensure numeric/categorical columns exist for transform function
                for c in num_cols:
                    X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce").fillna(X_raw[c].median())
                for c in cat_cols:
                    X_raw[c] = X_raw[c].astype(str).fillna("NA")
                X_proc = preproc.transform(X_raw[num_cols + cat_cols if len(cat_cols)>0 else num_cols])

                # optional balancing
                use_smt = st.checkbox("Apply SMOTETomek balancing if available", value=False)
                if use_smt and SMOTETomek is not None:
                    try:
                        smt = SMOTETomek(random_state=SEED)
                        X_bal, y_bal = smt.fit_resample(X_proc, y_enc)
                        st.write("SMOTETomek applied. New shape:", X_bal.shape)
                    except Exception as e:
                        st.warning("SMOTETomek failed; using original data.")
                        X_bal, y_bal = X_proc, y_enc
                else:
                    X_bal, y_bal = X_proc, y_enc

                # safe split
                X_train, X_test, y_train, y_test = safe_train_test_split(pd.DataFrame(X_bal), pd.Series(y_bal), test_size=0.25, random_state=SEED)
                # convert to numpy arrays
                X_train = np.asarray(X_train)
                X_test = np.asarray(X_test)
                y_train = np.asarray(y_train)
                y_test = np.asarray(y_test)

                st.info("Training ANN (baseline)...")
                t0 = time.time()
                try:
                    acc_ann, ann_model = train_ann_model(X_train, y_train, X_test, y_test)
                except Exception as e:
                    st.error(f"ANN training failed: {e}")
                    acc_ann, ann_model = 0.0, None

                st.info("Training ANN + GA (lightweight feature selection)...")
                try:
                    acc_ga, ann_ga_model, selected_idx = train_ann_ga_model(X_train, y_train, X_test, y_test)
                except Exception as e:
                    st.error(f"ANN+GA training failed: {e}")
                    acc_ga, ann_ga_model, selected_idx = 0.0, None, []

                st.info("Training Hybrid ANN -> XGBoost...")
                if XGBClassifier is None:
                    st.warning("xgboost not installed — hybrid skipped.")
                    acc_hybrid, ann_hybrid, xgb_hybrid = 0.0, None, None
                else:
                    try:
                        acc_hybrid, ann_hybrid, xgb_hybrid = train_hybrid_model(X_train, y_train, X_test, y_test)
                    except Exception as e:
                        st.error(f"Hybrid training failed: {e}")
                        acc_hybrid, ann_hybrid, xgb_hybrid = 0.0, None, None

                t_elapsed = time.time() - t0

                # show comparisons
                st.markdown("#### Model accuracies")
                st.write(f"ANN (baseline): **{acc_ann*100:.2f}%**")
                st.write(f"ANN + GA: **{acc_ga*100:.2f}%**")
                st.write(f"Hybrid ANN→XGBoost: **{acc_hybrid*100:.2f}%**")
                best_acc = max(acc_ann, acc_ga, acc_hybrid)
                if best_acc == acc_ann:
                    best_name = "ANN"
                    best_model_obj = ann_model
                elif best_acc == acc_ga:
                    best_name = "ANN+GA"
                    best_model_obj = ann_ga_model
                else:
                    best_name = "Hybrid ANN->XGBoost"
                    best_model_obj = {"ann": ann_hybrid, "xgb": xgb_hybrid} if xgb_hybrid is not None else None

                st.success(f"Training finished in {t_elapsed:.1f}s — Best: {best_name} ({best_acc*100:.2f}%)")

                # store artifacts
                artifacts = {
                    "preproc": preproc,
                    "num_cols": num_cols,
                    "cat_cols": cat_cols,
                    "target_le": target_le,
                    "best_name": best_name,
                    "best_model": best_model_obj,
                    "ann_model": ann_model,
                    "ann_ga_model": ann_ga_model,
                    "ann_ga_selected_idx": selected_idx,
                    "hybrid_ann": ann_hybrid,
                    "hybrid_xgb": xgb_hybrid
                }
                ok = save_artifacts(MODEL_FILE, artifacts)
                st.write("Saved artifacts:", ok)
                # store in session
                st.session_state["artifacts"] = artifacts

                # confusion matrix for best if possible
                try:
                    if best_name == "Hybrid ANN->XGBoost" and xgb_hybrid is not None:
                        ypred = xgb_hybrid.predict(ann_hybrid.predict_proba(X_test))
                    elif best_name == "ANN+GA" and ann_ga_model is not None and len(selected_idx)>0:
                        ypred = ann_ga_model.predict(X_test[:, selected_idx])
                    elif best_name == "ANN" and ann_model is not None:
                        ypred = ann_model.predict(X_test)
                    else:
                        ypred = None
                    if ypred is not None:
                        cm = confusion_matrix(y_test, ypred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        st.pyplot(fig)
                except Exception:
                    pass

st.markdown("---")

# Section: Prediction (Manual)
st.markdown("### 3) Manual Prediction (single sample)")
art = st.session_state.get("artifacts", None) or load_artifacts(MODEL_FILE)
if art is None:
    st.info("Train models and save artifacts first (use section 2).")
else:
    preproc = art["preproc"]
    num_cols = art["num_cols"]
    cat_cols = art["cat_cols"]
    target_le = art["target_le"]
    best_name = art["best_name"]
    best_model = art["best_model"]

    # Build inputs for features
    df_template = st.session_state.get("df")
    if df_template is not None:
        feat_cols = [c for c in df_template.columns if c != "Sleep Disorder"]
        st.write("Enter values for each feature (leave blank to use median/mode defaults):")
        inputs = {}
        cols_ui = st.columns(2)
        for i, c in enumerate(feat_cols):
            if c in num_cols:
                default = float(df_template[c].median()) if pd.api.types.is_numeric_dtype(df_template[c]) else 0.0
                inputs[c] = cols_ui[i%2].number_input(c, value=default)
            else:
                default = str(df_template[c].mode().iloc[0]) if c in df_template.columns and not df_template[c].mode().empty else "NA"
                inputs[c] = cols_ui[i%2].text_input(c, value=default)

        if st.button("Run manual prediction"):
            sample = pd.DataFrame([inputs])
            try:
                Xs = transform_with_preproc(sample, preproc, num_cols, cat_cols)
                if best_name == "Hybrid ANN->XGBoost":
                    ann_m = art.get("hybrid_ann")
                    xgb_m = art.get("hybrid_xgb")
                    if ann_m is None or xgb_m is None:
                        st.error("Hybrid models not available.")
                    else:
                        pred = xgb_m.predict(ann_m.predict_proba(Xs))[0]
                elif best_name == "ANN+GA":
                    ann_ga = art.get("ann_ga_model")
                    sel_idx = art.get("ann_ga_selected_idx", [])
                    if ann_ga is None or len(sel_idx) == 0:
                        st.error("ANN+GA model not available.")
                    else:
                        pred = ann_ga.predict(Xs[:, sel_idx])[0]
                else:
                    annm = art.get("ann_model")
                    if annm is None:
                        st.error("ANN model not available.")
                    else:
                        pred = annm.predict(Xs)[0]

                # decode label
                if target_le is not None:
                    pred_label = target_le.inverse_transform([int(pred)])[0]
                else:
                    pred_label = pred
                st.success(f"Predicted Sleep Disorder: {pred_label}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")

# Section: Prediction (Bulk)
st.markdown("### 4) Bulk Prediction (CSV)")
if art is None:
    st.info("Train and save artifacts first.")
else:
    upload_pred = st.file_uploader("Upload CSV for bulk prediction (features only or same columns)", type=["csv"], key="bulk_pred")
    if upload_pred is not None:
        try:
            newdf = pd.read_csv(upload_pred)
            # drop target if present
            if "Sleep Disorder" in newdf.columns:
                newdf = newdf.drop(columns=["Sleep Disorder"])
            # transform
            Xnew = transform_with_preproc(newdf, preproc, num_cols, cat_cols)
            # predict using best
            if best_name == "Hybrid ANN->XGBoost":
                ann_m = art.get("hybrid_ann")
                xgb_m = art.get("hybrid_xgb")
                preds = xgb_m.predict(ann_m.predict_proba(Xnew))
            elif best_name == "ANN+GA":
                ann_ga = art.get("ann_ga_model")
                sel_idx = art.get("ann_ga_selected_idx", [])
                preds = ann_ga.predict(Xnew[:, sel_idx]) if ann_ga is not None and len(sel_idx)>0 else np.array([None]*len(Xnew))
            else:
                annm = art.get("ann_model")
                preds = annm.predict(Xnew) if annm is not None else np.array([None]*len(Xnew))

            if target_le is not None:
                try:
                    labels = target_le.inverse_transform(preds.astype(int))
                except Exception:
                    labels = preds
            else:
                labels = preds
            out = newdf.copy()
            out["Predicted_Sleep_Disorder"] = labels
            st.dataframe(out.head(50))
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", csv_bytes, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Bulk prediction failed: {e}")

st.markdown("---")

# Section: Interpretability
st.markdown("### 5) Interpretability — Permutation Feature Importance (grouped by original features)")
if art is None:
    st.info("Train and save artifacts first.")
else:
    try:
        # Prepare dataset for importance (use original df)
        df_full = st.session_state.get("df")
        if df_full is None:
            st.warning("Dataset not available in session.")
        else:
            Xraw = df_full.drop(columns=["Sleep Disorder"])
            yraw = df_full["Sleep Disorder"]
            # transform all
            Xproc_all = transform_with_preproc(Xraw, preproc, num_cols, cat_cols)
            # prepare estimator wrapper depending on best model
            if art["best_name"] == "Hybrid ANN->XGBoost" and art.get("hybrid_ann") is not None and art.get("hybrid_xgb") is not None:
                class HybridWrapper:
                    def __init__(self, ann_model, xgb_model):
                        self.ann = ann_model
                        self.xgb = xgb_model
                    def predict(self, Xp):
                        return self.xgb.predict(self.ann.predict_proba(Xp))
                estimator = HybridWrapper(art["hybrid_ann"], art["hybrid_xgb"])
            elif art["best_name"] == "ANN+GA" and art.get("ann_ga_model") is not None:
                estimator = art["ann_ga_model"]
            else:
                estimator = art.get("ann_model") or art.get("ann_ga_model") or None

            if estimator is None:
                st.warning("No estimator available for permutation importance.")
            else:
                # encode target if needed
                y_enc = yraw
                if art.get("target_le") is not None:
                    y_enc = art["target_le"].transform(yraw.astype(str))
                st.info("Computing permutation importance (this may take a few seconds)...")
                res = permutation_importance(estimator, Xproc_all, y_enc, n_repeats=8, random_state=SEED, n_jobs=1)
                importances = res.importances_mean
                # reconstruct processed column names (num + cat__val)
                proc_names = []
                if len(num_cols) > 0:
                    proc_names.extend(num_cols)
                if len(cat_cols) > 0:
                    ohe = preproc.named_transformers_.get("cat")
                    if ohe is None:
                        # When cat transformer is inside ColumnTransformer, retrieve it differently
                        try:
                            ct = preproc
                            # find categories stored in the OneHotEncoder
                            ohe_obj = None
                            for name, trans, cols in ct.transformers_:
                                if name == "cat":
                                    ohe_obj = trans.named_steps["ohe"]
                                    cat_list = cols
                                    break
                            if ohe_obj is not None:
                                for i, col in enumerate(cat_cols):
                                    cats = ohe_obj.categories_[i]
                                    proc_names.extend([f"{col}__{v}" for v in cats])
                        except Exception:
                            # fallback: name columns generically
                            proc_names.extend([f"cat_{i}" for i in range(len(importances)-len(num_cols))])
                    else:
                        # if cat transformer pipeline exists
                        pass
                # If proc_names length mismatches, fallback to numeric indexes
                if len(proc_names) != len(importances):
                    proc_names = [f"feat_{i}" for i in range(len(importances))]
                imp_df = pd.DataFrame({"proc_col": proc_names, "importance": importances})
                # group by original col (split on '__')
                imp_df["orig"] = imp_df["proc_col"].apply(lambda x: x.split("__")[0] if "__" in x else x)
                agg = imp_df.groupby("orig")["importance"].sum().reset_index().sort_values("importance", ascending=False)
                st.dataframe(agg)
                fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(agg))))
                sns.barplot(x="importance", y="orig", data=agg, ax=ax)
                ax.set_xlabel("Permutation importance (mean)")
                ax.set_ylabel("Feature")
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Interpretability error: {e}")

# EOF
