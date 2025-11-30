# app.py
"""
Single-file Streamlit app (executable)
Features:
- Top horizontal navigation bar (streamlit-option-menu)
- Upload dataset (expects target column exactly "Sleep Disorder")
- Train 3 models: ANN, ANN+GA (feature-selection), Hybrid (ANN -> XGBoost)
- Safe preprocessing (OneHotEncoder compatibility), safe train/test split
- Manual & Bulk prediction
- Permutation feature importance (group OHE by original feature)
- Save / load artifacts to disk
- Designed to be robust to common dataset issues
Requirements:
pip install streamlit pandas numpy scikit-learn xgboost streamlit-option-menu imbalanced-learn matplotlib seaborn
(Installing imbalanced-learn and xgboost is optional — app skips gracefully if missing.)
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, time, random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
random.seed(42)
np.random.seed(42)

# optional libs
try:
    from streamlit_option_menu import option_menu
except Exception:
    option_menu = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from imblearn.combine import SMOTETomek
except Exception:
    SMOTETomek = None

# constants
SEED = 42
ARTIFACT_PATH = "artifacts.pkl"

st.set_page_config(page_title="Sleep Disorder Classifier", layout="wide")

# ---------------------------
# helper: compatible OneHotEncoder
# ---------------------------
def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# ---------------------------
# helper: safe train_test_split
# ---------------------------
def safe_split(X, y, test_size=0.25, random_state=SEED):
    y_ser = pd.Series(y)
    if y_ser.nunique() < 2:
        st.warning("Target has only one class — stratified split disabled.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    counts = y_ser.value_counts()
    if any(counts < 3):
        st.warning("Some classes have <3 samples — stratified split disabled to avoid error.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

# ---------------------------
# Preprocessing helpers
# ---------------------------
def build_preprocessor_from_df(df, target_col="Sleep Disorder"):
    # decide numeric & categorical
    X = df.drop(columns=[target_col])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    # remove target if mistakenly included
    if target_col in num_cols: num_cols.remove(target_col)
    if target_col in cat_cols: cat_cols.remove(target_col)
    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", Pipeline([("scaler", StandardScaler())]), num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", Pipeline([("ohe", make_onehot())]), cat_cols))
    if len(transformers) == 0:
        ct = ColumnTransformer([("none", "passthrough", X.columns.tolist())])
    else:
        ct = ColumnTransformer(transformers, remainder="drop")
    return ct, num_cols, cat_cols

def fit_preprocessor(df, target_col="Sleep Disorder"):
    ct, num_cols, cat_cols = build_preprocessor_from_df(df, target_col)
    X = df.drop(columns=[target_col])
    # coerce numeric columns
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")
    ct.fit(X[num_cols + cat_cols if len(cat_cols)>0 else num_cols])
    return ct, num_cols, cat_cols

def transform_df(df_new, preproc, num_cols, cat_cols):
    # ensure columns present
    df = df_new.copy()
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "NA"
    # coerce types
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("NA")
    cols = num_cols + (cat_cols if len(cat_cols)>0 else [])
    return preproc.transform(df[cols])

# ---------------------------
# GA feature selection (fixed; returns selected indices)
# ---------------------------
def ga_feature_selection(X, y, generations=6, pop_size=10, mutation_rate=0.15, timeout=45):
    """
    Lightweight GA that evolves binary masks selecting features.
    Returns list of selected column indices.
    """
    rng = np.random.RandomState(SEED)
    n_feat = X.shape[1]
    if n_feat <= 2:
        return list(range(n_feat))
    # initialize population (binary masks)
    population = rng.randint(0,2,size=(pop_size,n_feat))
    # ensure minimum features selected
    for i in range(pop_size):
        if population[i].sum() < 2:
            idx = rng.choice(n_feat, 2, replace=False)
            population[i, idx] = 1
    start = time.time()
    def fitness(mask):
        if mask.sum() < 2:
            return 0.0
        sel = np.where(mask==1)[0]
        try:
            Xt, Xv, yt, yv = train_test_split(X[:, sel], y, test_size=0.25, random_state=SEED)
        except Exception:
            Xt, Xv, yt, yv = train_test_split(X[:, sel], y, test_size=0.25)
        try:
            clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=SEED)
            clf.fit(Xt, yt)
            p = clf.predict(Xv)
            return accuracy_score(yv, p)
        except Exception:
            return 0.0
    best_mask = None
    best_score = -1.0
    for gen in range(generations):
        scores = np.array([fitness(m) for m in population])
        order = scores.argsort()[::-1]
        # keep top 40%
        keep_n = max(2, int(pop_size*0.4))
        parents = population[order[:keep_n]]
        # track best
        if scores.max() > best_score:
            best_score = scores.max()
            best_mask = population[scores.argmax()].copy()
        # breed
        new_pop = parents.tolist()
        while len(new_pop) < pop_size:
            a,b = parents[rng.randint(0,len(parents))], parents[rng.randint(0,len(parents))]
            # single-point crossover
            pt = rng.randint(1, n_feat-1)
            child = np.concatenate([a[:pt], b[pt:]])
            # mutation
            if rng.rand() < mutation_rate:
                m_idx = rng.randint(0, n_feat)
                child[m_idx] = 1 - child[m_idx]
            # ensure >=2 features
            if child.sum() < 2:
                idxs = rng.choice(n_feat, 2, replace=False)
                child[idxs] = 1
            new_pop.append(child)
        population = np.array(new_pop)
        if time.time() - start > timeout:
            break
    if best_mask is None:
        best_mask = population[0]
    selected = np.where(best_mask==1)[0].tolist()
    return selected

# ---------------------------
# model training functions
# ---------------------------
def train_ann(X_train, y_train, X_test, y_test):
    ann = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=400, random_state=SEED)
    ann.fit(X_train, y_train)
    p = ann.predict(X_test)
    return accuracy_score(y_test, p), ann

def train_ann_ga(X_train, y_train, X_test, y_test):
    sel = ga_feature_selection(X_train, y_train, generations=6, pop_size=10, timeout=45)
    if len(sel) == 0:
        sel = list(range(min(2, X_train.shape[1])))
    Xtr, Xte = X_train[:, sel], X_test[:, sel]
    ann = MLPClassifier(hidden_layer_sizes=(96,48), max_iter=400, random_state=SEED)
    ann.fit(Xtr, y_train)
    p = ann.predict(Xte)
    return accuracy_score(y_test, p), ann, sel

def train_hybrid(X_train, y_train, X_test, y_test):
    if XGBClassifier is None:
        raise RuntimeError("xgboost not installed")
    ann = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=SEED)
    ann.fit(X_train, y_train)
    # use ann probabilities as features
    tr_proba = ann.predict_proba(X_train)
    te_proba = ann.predict_proba(X_test)
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric="mlogloss", random_state=SEED)
    xgb.fit(tr_proba, y_train)
    p = xgb.predict(te_proba)
    return accuracy_score(y_test, p), ann, xgb

# ---------------------------
# Save / Load artifacts
# ---------------------------
def save_artifacts(path, obj):
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

def load_artifacts(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Load failed: {e}")
            return None
    return None

# ---------------------------
# Top nav (horizontal) using streamlit-option-menu if available
# ---------------------------
if option_menu is not None:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Upload", "Train", "Predict Manual", "Predict Bulk", "Interpret"],
        icons=["house", "cloud-upload", "cpu", "person-lines-fill", "file-earmark-spreadsheet", "bar-chart"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0px", "background-color": "#f0f8ff"},
            "icon": {"color": "#0b5ed7", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px 8px"},
            "nav-link-selected": {"background-color": "#0b5ed7", "color": "white"},
        }
    )
else:
    # fallback to sidebar radio
    st.sidebar.warning("Install streamlit-option-menu for top navbar: pip install streamlit-option-menu")
    selected = st.sidebar.radio("Navigation", ["Home", "Upload", "Train", "Predict Manual", "Predict Bulk", "Interpret"])

# ---------------------------
# Page: Home
# ---------------------------
if selected == "Home":
    st.title("😴 Sleep Disorder Classification")
    st.markdown(
        """
        #### Why sleep matters
        - Sleep is essential for memory, mood, immune function and metabolic health.
        - Common sleep disorders: *Insomnia, Sleep Apnea, Narcolepsy, Restless Leg Syndrome*.
        - Early detection improves outcomes.

        Use the navigation bar to upload your dataset, train models, make predictions, and inspect feature importance.
        """
    )
    st.info("Target column must be named exactly: **Sleep Disorder**")

# ---------------------------
# Page: Upload
# ---------------------------
elif selected == "Upload":
    st.header("Upload dataset (CSV)")
    uploaded = st.file_uploader("Upload CSV (features + target 'Sleep Disorder')", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            # convenience blood pressure parsing
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
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        if "df" in st.session_state and st.session_state["df"] is not None:
            st.write("Dataset already loaded in session:")
            st.dataframe(st.session_state["df"].head(6))

# ---------------------------
# Page: Train
# ---------------------------
elif selected == "Train":
    st.header("Train & Compare Models")
    df = st.session_state.get("df", None)
    if df is None:
        st.warning("Upload dataset first.")
    else:
        if "Sleep Disorder" not in df.columns:
            st.error("Target column 'Sleep Disorder' not found in dataset.")
        else:
            st.write("Dataset shape:", df.shape)
            if st.button("Fit Preprocessor"):
                with st.spinner("Fitting preprocessor..."):
                    preproc, num_cols, cat_cols = fit_preprocessor(df, "Sleep Disorder")
                    st.session_state["preproc"] = preproc
                    st.session_state["num_cols"] = num_cols
                    st.session_state["cat_cols"] = cat_cols
                    st.success("Preprocessor fitted. Numeric cols: %s | Categorical cols: %s" % (num_cols, cat_cols))

            if "preproc" in st.session_state:
                preproc = st.session_state["preproc"]
                num_cols = st.session_state["num_cols"]
                cat_cols = st.session_state["cat_cols"]

                if st.button("Train 3 Models (ANN, ANN+GA, Hybrid)"):
                    with st.spinner("Training — this may take some time depending on dataset and xgboost availability..."):
                        # build X,y processed
                        Xraw = df.drop(columns=["Sleep Disorder"])
                        yraw = df["Sleep Disorder"]
                        # encode target
                        le_target = LabelEncoder()
                        y_enc = le_target.fit_transform(yraw.astype(str))
                        st.session_state["target_le"] = le_target

                        # ensure numeric/cat presence
                        for c in num_cols:
                            if c not in Xraw.columns:
                                Xraw[c] = 0
                            Xraw[c] = pd.to_numeric(Xraw[c], errors="coerce").fillna(Xraw[c].median())
                        for c in cat_cols:
                            if c not in Xraw.columns:
                                Xraw[c] = "NA"
                            Xraw[c] = Xraw[c].astype(str).fillna("NA")

                        Xproc = preproc.transform(Xraw[num_cols + cat_cols if len(cat_cols)>0 else num_cols])

                        # optional balancing
                        use_smt = st.checkbox("Apply SMOTETomek balancing (if installed)", value=False)
                        if use_smt and SMOTETomek is not None:
                            try:
                                smt = SMOTETomek(random_state=SEED)
                                Xbal, ybal = smt.fit_resample(Xproc, y_enc)
                                st.write("SMOTETomek applied. New shape:", Xbal.shape)
                            except Exception as e:
                                st.warning("SMOTETomek failed: " + str(e))
                                Xbal, ybal = Xproc, y_enc
                        else:
                            Xbal, ybal = Xproc, y_enc

                        # safe split
                        X_train, X_test, y_train, y_test = safe_split(pd.DataFrame(Xbal), pd.Series(ybal), test_size=0.25)
                        X_train = np.asarray(X_train); X_test = np.asarray(X_test)
                        y_train = np.asarray(y_train); y_test = np.asarray(y_test)

                        # train ANN
                        try:
                            acc_ann, ann_model = train_ann(X_train, y_train, X_test, y_test)
                        except Exception as e:
                            st.error("ANN failed: " + str(e)); acc_ann, ann_model = 0.0, None

                        # train ANN+GA
                        try:
                            acc_ga, ann_ga, sel_idx = train_ann_ga(X_train, y_train, X_test, y_test)
                        except Exception as e:
                            st.error("ANN+GA failed: " + str(e)); acc_ga, ann_ga, sel_idx = 0.0, None, []

                        # train hybrid
                        if XGBClassifier is not None:
                            try:
                                acc_hyb, ann_h, xgb_h = train_hybrid(X_train, y_train, X_test, y_test)
                            except Exception as e:
                                st.error("Hybrid failed: " + str(e)); acc_hyb, ann_h, xgb_h = 0.0, None, None
                        else:
                            st.info("xgboost not installed — skipping hybrid")
                            acc_hyb, ann_h, xgb_h = 0.0, None, None

                        st.markdown("#### Accuracies")
                        st.write(f"ANN: **{acc_ann*100:.2f}%**")
                        st.write(f"ANN + GA: **{acc_ga*100:.2f}%**")
                        st.write(f"Hybrid ANN→XGBoost: **{acc_hyb*100:.2f}%**")

                        best_acc = max(acc_ann, acc_ga, acc_hyb)
                        if best_acc == acc_ann:
                            best_name = "ANN"
                            best_model = ann_model
                        elif best_acc == acc_ga:
                            best_name = "ANN+GA"
                            best_model = ann_ga
                        else:
                            best_name = "Hybrid"
                            best_model = {"ann": ann_h, "xgb": xgb_h}

                        st.success(f"Best: {best_name} ({best_acc*100:.2f}%)")

                        # save artifacts
                        artifacts = {
                            "preproc": preproc,
                            "num_cols": num_cols,
                            "cat_cols": cat_cols,
                            "target_le": le_target,
                            "best_name": best_name,
                            "best_model": best_model,
                            "ann_model": ann_model,
                            "ann_ga_model": ann_ga,
                            "ann_ga_sel": sel_idx,
                            "hybrid_ann": ann_h,
                            "hybrid_xgb": xgb_h
                        }
                        ok = save_artifacts(ARTIFACT_PATH, artifacts)
                        st.write("Saved artifacts:", ok)
                        st.session_state["artifacts"] = artifacts

# ---------------------------
# Page: Predict Manual
# ---------------------------
elif selected == "Predict Manual":
    st.header("Manual Prediction (single sample)")
    artifacts = st.session_state.get("artifacts", None) or load_artifacts(ARTIFACT_PATH)
    if artifacts is None:
        st.warning("Train models and save artifacts first.")
    else:
        df_template = st.session_state.get("df")
        if df_template is None:
            st.error("Upload dataset first.")
        else:
            features = [c for c in df_template.columns if c != "Sleep Disorder"]
            num_cols = artifacts["num_cols"]
            cat_cols = artifacts["cat_cols"]
            inputs = {}
            cols_ui = st.columns(2)
            for i, f in enumerate(features):
                if f in num_cols:
                    default = float(df_template[f].median()) if pd.api.types.is_numeric_dtype(df_template[f]) else 0.0
                    inputs[f] = cols_ui[i%2].number_input(f, value=default)
                else:
                    default = str(df_template[f].mode().iloc[0]) if (f in df_template.columns and not df_template[f].mode().empty) else "NA"
                    inputs[f] = cols_ui[i%2].text_input(f, value=default)
            if st.button("Predict"):
                sample = pd.DataFrame([inputs])
                try:
                    Xs = transform_df(sample, artifacts["preproc"], num_cols, cat_cols)
                    best_name = artifacts["best_name"]
                    if best_name == "Hybrid":
                        ann_m = artifacts.get("hybrid_ann"); xgb_m = artifacts.get("hybrid_xgb")
                        if ann_m is None or xgb_m is None:
                            st.error("Hybrid model not available.")
                        else:
                            pred = xgb_m.predict(ann_m.predict_proba(Xs))[0]
                    elif best_name == "ANN+GA":
                        ann_ga = artifacts.get("ann_ga_model"); sel = artifacts.get("ann_ga_sel", [])
                        if ann_ga is None or len(sel)==0:
                            st.error("ANN+GA not available.")
                        else:
                            pred = ann_ga.predict(Xs[:, sel])[0]
                    else:
                        ann_m = artifacts.get("ann_model")
                        if ann_m is None:
                            st.error("ANN not available.")
                        else:
                            pred = ann_m.predict(Xs)[0]
                    # decode
                    target_le = artifacts.get("target_le", None)
                    if target_le is not None:
                        try:
                            label = target_le.inverse_transform([int(pred)])[0]
                        except Exception:
                            label = pred
                    else:
                        label = pred
                    st.success(f"Predicted Sleep Disorder: {label}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ---------------------------
# Page: Predict Bulk
# ---------------------------
elif selected == "Predict Bulk":
    st.header("Bulk Prediction (CSV)")
    artifacts = st.session_state.get("artifacts", None) or load_artifacts(ARTIFACT_PATH)
    if artifacts is None:
        st.warning("Train models and save artifacts first.")
    else:
        upload = st.file_uploader("Upload CSV for prediction", type=["csv"])
        if upload is not None:
            try:
                newdf = pd.read_csv(upload)
                if "Sleep Disorder" in newdf.columns:
                    newdf = newdf.drop(columns=["Sleep Disorder"])
                Xnew = transform_df(newdf, artifacts["preproc"], artifacts["num_cols"], artifacts["cat_cols"])
                best_name = artifacts["best_name"]
                if best_name == "Hybrid":
                    ann_m = artifacts.get("hybrid_ann"); xgb_m = artifacts.get("hybrid_xgb")
                    preds = xgb_m.predict(ann_m.predict_proba(Xnew)) if ann_m is not None and xgb_m is not None else np.array([None]*len(Xnew))
                elif best_name == "ANN+GA":
                    ann_ga = artifacts.get("ann_ga_model"); sel = artifacts.get("ann_ga_sel", [])
                    preds = ann_ga.predict(Xnew[:, sel]) if ann_ga is not None and len(sel)>0 else np.array([None]*len(Xnew))
                else:
                    ann_m = artifacts.get("ann_model")
                    preds = ann_m.predict(Xnew) if ann_m is not None else np.array([None]*len(Xnew))
                # decode
                tle = artifacts.get("target_le", None)
                if tle is not None:
                    try:
                        labels = tle.inverse_transform(preds.astype(int))
                    except Exception:
                        labels = preds
                else:
                    labels = preds
                out = newdf.copy()
                out["Predicted_Sleep_Disorder"] = labels
                st.dataframe(out.head(100))
                st.download_button("Download predictions CSV", out.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
            except Exception as e:
                st.error("Bulk predict failed: " + str(e))

# ---------------------------
# Page: Interpret
# ---------------------------
elif selected == "Interpret":
    st.header("Permutation Feature Importance (grouped)")
    artifacts = st.session_state.get("artifacts", None) or load_artifacts(ARTIFACT_PATH)
    df = st.session_state.get("df", None)
    if artifacts is None or df is None:
        st.warning("Train and upload dataset first.")
    else:
        try:
            Xraw = df.drop(columns=["Sleep Disorder"])
            yraw = df["Sleep Disorder"]
            tle = artifacts.get("target_le", None)
            y_enc = tle.transform(yraw.astype(str)) if tle is not None else yraw.values
            Xproc = transform_df(Xraw, artifacts["preproc"], artifacts["num_cols"], artifacts["cat_cols"])
            # wrapper estimator
            best_name = artifacts["best_name"]
            if best_name == "Hybrid" and artifacts.get("hybrid_ann") is not None and artifacts.get("hybrid_xgb") is not None:
                class Hybrid:
                    def __init__(self,a,b): self.a=a; self.b=b
                    def predict(self,Xp): return self.b.predict(self.a.predict_proba(Xp))
                estimator = Hybrid(artifacts["hybrid_ann"], artifacts["hybrid_xgb"])
            elif best_name == "ANN+GA" and artifacts.get("ann_ga_model") is not None:
                estimator = artifacts["ann_ga_model"]
            else:
                estimator = artifacts.get("ann_model")
            if estimator is None:
                st.error("No trained estimator available for importance.")
            else:
                st.info("Computing permutation importance (may take a few seconds)...")
                res = permutation_importance(estimator, Xproc, y_enc, n_repeats=8, random_state=SEED, n_jobs=1)
                importances = res.importances_mean
                # reconstruct processed column names
                proc_names = []
                proc_names.extend(artifacts["num_cols"])
                if len(artifacts["cat_cols"])>0:
                    # try to pull categories from OneHotEncoder inside ColumnTransformer
                    try:
                        ohe = artifacts["preproc"].named_transformers_["cat"].named_steps["ohe"]
                        for i, col in enumerate(artifacts["cat_cols"]):
                            cats = ohe.categories_[i]
                            proc_names.extend([f"{col}__{c}" for c in cats])
                    except Exception:
                        # fallback: generic names
                        proc_names.extend([f"cat_{i}" for i in range(len(importances)-len(artifacts["num_cols"]))])
                if len(proc_names) != len(importances):
                    proc_names = [f"feat_{i}" for i in range(len(importances))]
                imp_df = pd.DataFrame({"proc_col": proc_names, "importance": importances})
                imp_df["orig"] = imp_df["proc_col"].apply(lambda x: x.split("__")[0] if "__" in x else x)
                agg = imp_df.groupby("orig")["importance"].sum().reset_index().sort_values("importance", ascending=False)
                st.dataframe(agg)
                fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(agg))))
                sns.barplot(x="importance", y="orig", data=agg, ax=ax)
                ax.set_xlabel("Permutation importance (mean)")
                ax.set_ylabel("Feature")
                st.pyplot(fig)
        except Exception as e:
            st.error("Interpretability failed: " + str(e))

# ---------------------------
# Footer note
# ---------------------------
st.markdown("<hr><div style='font-size:12px;color:#666;'>Note: this app expects a dataset with a target column named exactly 'Sleep Disorder'. If your CSV uses a different name, rename it before uploading.</div>", unsafe_allow_html=True)
