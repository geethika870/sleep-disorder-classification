# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# Optional imports (graceful fallback)
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import shap
except Exception:
    shap = None

# TensorFlow (ANN) optional
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    tf = None
    keras = None

# Page config
st.set_page_config(page_title="Sleep Disorder Classifier", layout="wide", initial_sidebar_state="expanded")

# Styling (colorful, no images)
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(90deg,#f7fbff,#eef7ff); }
    .title {font-size:36px; font-weight:700; color:#083d77; padding-bottom:6px;}
    .subtitle {color:#083d77; font-size:14px;}
    .card {background: linear-gradient(180deg,#ffffff,#f4fbff); padding:16px; border-radius:12px; box-shadow: 0 6px 18px rgba(8,61,119,0.08);}
    .section {padding:12px; border-radius:10px; background: rgba(255,255,255,0.6);}
    </style>
    """, unsafe_allow_html=True
)

# -------------------------
# SESSION STATE INIT
# -------------------------
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None
if 'feature_columns' not in st.session_state:
    st.session_state['feature_columns'] = None
if 'target_column' not in st.session_state:
    st.session_state['target_column'] = None
if 'encoders' not in st.session_state:
    st.session_state['encoders'] = {}
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None
if 'trained_model_type' not in st.session_state:
    st.session_state['trained_model_type'] = None
if 'label_encoder_target' not in st.session_state:
    st.session_state['label_encoder_target'] = None
if 'best_score' not in st.session_state:
    st.session_state['best_score'] = None
if 'best_model_info' not in st.session_state:
    st.session_state['best_model_info'] = None

# -------------------------
# HELPER FUNCTIONS
# -------------------------
@st.cache_data
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        st.warning("⚠ Stratified split failed (likely too few samples in some classes). Using random split instead.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_for_model(df: pd.DataFrame, target_col: str):
    """
    Basic preprocessing:
      - drop fully empty rows
      - fill numeric with median, categorical with mode
      - label-encode categorical features (store encoders)
      - scale numeric features with StandardScaler
    Returns: X (np.array), y (np.array), encoders, scaler, feature_columns
    """
    df = df.copy()
    df = df.dropna(how='all')
    # separate
    if target_col not in df.columns:
        raise ValueError("Target column not found in dataset.")
    y_series = df[target_col].copy()
    X_df = df.drop(columns=[target_col])

    # fill numeric & categorical
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()

    for c in num_cols:
        X_df[c] = X_df[c].fillna(X_df[c].median())
    for c in cat_cols:
        X_df[c] = X_df[c].fillna(X_df[c].mode().iloc[0] if not X_df[c].mode().empty else "NA")

    # encode categorical features
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        X_df[c] = le.fit_transform(X_df[c].astype(str))
        encoders[c] = le

    # encode target if categorical
    target_le = None
    if y_series.dtype == 'object' or y_series.dtype.name == 'category':
        target_le = LabelEncoder()
        y = target_le.fit_transform(y_series.astype(str))
    else:
        y = y_series.values

    # scale numeric (after encoding cat => all numeric)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    return X_scaled, y, encoders, scaler, X_df.columns.tolist(), target_le

def plot_confusion(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if labels is not None:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    st.pyplot(fig)

# -------------------------
# GA + ANN utilities (lightweight)
# -------------------------
def build_keras_model(input_dim, n_layers=1, n_neurons=16, activation='relu', lr=0.001, dropout=0.0, multiclass=False, n_classes=2):
    if keras is None:
        raise RuntimeError("TensorFlow/Keras not installed.")
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for _ in range(n_layers):
        model.add(layers.Dense(n_neurons, activation=activation))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout))
    if multiclass:
        model.add(layers.Dense(n_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
    else:
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model

def simple_ga_tune_ann(X_train, y_train, X_val, y_val, generations=4, pop_size=6, multiclass=False, random_state=42):
    """
    Small GA: searches small hyperparam space for demo purposes.
    Returns: trained final keras model, best_val_score, best_params
    """
    np.random.seed(random_state)
    param_space = {
        "n_layers": [1,2,3],
        "n_neurons": [8,16,32,64,128],
        "lr": [1e-4, 3e-4, 1e-3, 3e-3],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "activation": ['relu', 'tanh']
    }

    def rand_ind():
        return {
            "n_layers": int(np.random.choice(param_space['n_layers'])),
            "n_neurons": int(np.random.choice(param_space['n_neurons'])),
            "lr": float(np.random.choice(param_space['lr'])),
            "dropout": float(np.random.choice(param_space['dropout'])),
            "activation": str(np.random.choice(param_space['activation']))
        }

    def fitness(ind):
        try:
            model = build_keras_model(input_dim=X_train.shape[1],
                                      n_layers=ind['n_layers'],
                                      n_neurons=ind['n_neurons'],
                                      activation=ind['activation'],
                                      lr=ind['lr'],
                                      dropout=ind['dropout'],
                                      multiclass=multiclass,
                                      n_classes=len(np.unique(y_train)) if multiclass else 2)
            # short training to evaluate
            epochs = 20
            bs = 32
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=bs, verbose=0)
            val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
            keras.backend.clear_session()
            return val_acc
        except Exception:
            return 0.0

    # init
    population = [rand_ind() for _ in range(pop_size)]
    best = (0.0, None)
    for gen in range(generations):
        st.sidebar.write(f"GA gen {gen+1}/{generations}")
        scored = []
        for ind in population:
            score = fitness(ind)
            scored.append((score, ind))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > best[0]:
            best = scored[0]
        # selection & breeding
        survivors = [ind for (_, ind) in scored[: max(2, pop_size//2)]]
        new_pop = survivors.copy()
        while len(new_pop) < pop_size:
            a, b = np.random.choice(survivors, 2, replace=False)
            child = {}
            for k in a.keys():
                child[k] = a[k] if np.random.rand() < 0.5 else b[k]
            # mutate
            if np.random.rand() < 0.25:
                key = np.random.choice(list(child.keys()))
                child[key] = rand_ind()[key]
            new_pop.append(child)
        population = new_pop

    # Train final model on train+val with best params
    best_params = best[1] or rand_ind()
    final_model = build_keras_model(input_dim=X_train.shape[1],
                                    n_layers=best_params['n_layers'],
                                    n_neurons=best_params['n_neurons'],
                                    activation=best_params['activation'],
                                    lr=best_params['lr'],
                                    dropout=best_params['dropout'],
                                    multiclass=multiclass,
                                    n_classes=len(np.unique(y_train)) if multiclass else 2)
    final_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]), epochs=40, batch_size=32, verbose=0)
    return final_model, best[0], best_params

# -------------------------
# NAVIGATION (no Home option)
# -------------------------
st.sidebar.title("Navigation")
nav = st.sidebar.radio("", ["Upload Dataset", "Train Model (ANN + GA)", "Manual Prediction", "Bulk Prediction", "Interpretability"], index=0)

# -------------------------
# DEFAULT HOMEPAGE (always shown by default at top)
# -------------------------
st.markdown('<div class="title">Sleep Disorder Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A focused app to train models (ANN+GA), predict sleep-disorder labels, and explain decisions — colorful UI, no images.</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("""
### Why sleep matters
- Sleep is crucial for memory, mood, immunity and cognitive performance.
- Sleep disorders (insomnia, sleep apnea, narcolepsy, restless legs, etc.) affect millions worldwide.
- Automated screening using machine learning helps triage high-risk individuals.

**Instructions (one-time):**
1. Go to **Upload Dataset** and upload your CSV (it will be stored in session and used everywhere).  
2. Then go to **Train Model (ANN + GA)** to train and compare models.  
3. Use **Manual Prediction** or **Bulk Prediction** to infer new samples.  
4. Use **Interpretability** for feature importance / SHAP explanations.

(Your uploaded dataset will not be asked again.)
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# PAGES
# -------------------------
# 1) Upload Dataset
if nav == "Upload Dataset":
    st.header("Upload dataset (CSV) — single upload used everywhere")
    uploaded = st.file_uploader("Upload your dataset CSV (one-time)", type=["csv"], key="main_upload")
    if uploaded is not None:
        try:
            df = read_csv(uploaded)
            st.session_state['dataset'] = df
            st.success("Dataset loaded and stored in session.")
            st.write("Shape:", df.shape)
            st.dataframe(df.head(10))
            # Ask target column
            cols = df.columns.tolist()
            col = st.selectbox("Select target (label) column", cols, index=len(cols)-1)
            st.session_state['target_column'] = col
            if st.button("Confirm dataset & target"):
                # preprocess now and store encoders & scaler & feature columns
                try:
                    Xs, y, encs, scaler, feat_cols, target_le = preprocess_for_model(df, col)
                    st.session_state['feature_columns'] = feat_cols
                    st.session_state['encoders'] = encs
                    st.session_state['scaler'] = scaler
                    st.session_state['label_encoder_target'] = target_le
                    st.success("Preprocessing complete and stored. You can now train the model.")
                except Exception as e:
                    st.error("Preprocessing failed: " + str(e))
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))
    else:
        if st.session_state['dataset'] is not None:
            st.info("Dataset already uploaded in this session. You can proceed to Train Model, Predict, or Interpretability.")
            st.write("Dataset shape:", st.session_state['dataset'].shape)
            st.dataframe(st.session_state['dataset'].head(5))
        else:
            st.info("Upload dataset to proceed. Once uploaded it will be used everywhere in this session.")

# 2) Train Model
elif nav == "Train Model (ANN + GA)":
    st.header("Train & Evaluate — compare RF / XGB / LGBM / GA-ANN")
    if st.session_state['dataset'] is None:
        st.warning("Upload a dataset first under 'Upload Dataset'. (It will be stored in session.)")
    else:
        df = st.session_state['dataset']
        target_col = st.session_state['target_column']
        st.write("Using target:", target_col)
        test_size = st.slider("Test set proportion", 0.1, 0.4, 0.2)
        random_state = int(st.number_input("Random seed", value=42, step=1))
        run = st.button("Run training & comparison")
        if run:
            with st.spinner("Preprocessing and splitting..."):
                Xs, y, encs, scaler, feat_cols, target_le = preprocess_for_model(df, target_col)
                st.session_state['encoders'] = encs
                st.session_state['scaler'] = scaler
                st.session_state['feature_columns'] = feat_cols
                st.session_state['label_encoder_target'] = target_le
                X_train, X_test, y_train, y_test = safe_train_test_split(Xs, y, test_size=test_size, random_state=random_state)

            results = {}
            # RandomForest
            st.write("Training RandomForest...")
            rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            acc_rf = accuracy_score(y_test, preds)
            results['RandomForest'] = (acc_rf, rf)
            st.write("RandomForest accuracy:", acc_rf)

            # XGBoost
            if xgb is not None:
                st.write("Training XGBoost...")
                xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
                xgb_clf.fit(X_train, y_train)
                preds = xgb_clf.predict(X_test)
                acc_xgb = accuracy_score(y_test, preds)
                results['XGBoost'] = (acc_xgb, xgb_clf)
                st.write("XGBoost accuracy:", acc_xgb)

            # LightGBM
            if lgb is not None:
                st.write("Training LightGBM...")
                lgb_clf = lgb.LGBMClassifier(random_state=random_state)
                lgb_clf.fit(X_train, y_train)
                preds = lgb_clf.predict(X_test)
                acc_lgb = accuracy_score(y_test, preds)
                results['LightGBM'] = (acc_lgb, lgb_clf)
                st.write("LightGBM accuracy:", acc_lgb)

            # GA + ANN (only if keras available)
            if keras is not None:
                st.write("Running GA tuner for ANN (lightweight)...")
                multiclass = len(np.unique(y_train)) > 2
                # split train into train+val for GA
                Xt, Xval, yt, yval = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train if len(np.unique(y_train))>1 else None)
                try:
                    ann_model, best_val, best_params = simple_ga_tune_ann(Xt, yt, Xval, yval, generations=4, pop_size=6, multiclass=multiclass, random_state=random_state)
                    # evaluate on test
                    if multiclass:
                        probs = ann_model.predict(X_test)
                        preds = probs.argmax(axis=1)
                    else:
                        probs = ann_model.predict(X_test).ravel()
                        preds = (probs >= 0.5).astype(int)
                    acc_ann = accuracy_score(y_test, preds)
                    results['GA-ANN'] = (acc_ann, ann_model)
                    st.write("GA-ANN test accuracy:", acc_ann)
                    st.write("GA best params:", best_params)
                except Exception as e:
                    st.error("GA-ANN failed: " + str(e))
            else:
                st.info("TensorFlow/Keras not installed — GA-ANN skipped.")

            # choose best
            best_name, (best_acc, best_mod) = max(results.items(), key=lambda x: x[1][0])
            st.success(f"Best model: {best_name} with accuracy {best_acc:.4f}")
            # store model & scaler in session
            st.session_state['trained_model'] = best_mod
            st.session_state['trained_model_type'] = best_name
            st.session_state['best_score'] = best_acc
            st.session_state['best_model_info'] = {'name': best_name, 'accuracy': float(best_acc)}
            # save to disk optionally
            try:
                joblib.dump({'model': best_mod, 'scaler': st.session_state['scaler'], 'type': best_name}, "best_model.joblib")
                st.write("Saved best_model.joblib in app folder.")
            except Exception:
                st.info("Could not save model to disk in this environment (permission).")

            # show classification report & confusion for best model
            st.subheader("Evaluation — Best model")
            if 'GA-ANN' in best_name and keras is not None:
                if len(np.unique(y_train)) > 2:
                    probs = best_mod.predict(X_test)
                    ypred = probs.argmax(axis=1)
                else:
                    probs = best_mod.predict(X_test).ravel()
                    ypred = (probs >= 0.5).astype(int)
            else:
                ypred = best_mod.predict(X_test)
            st.text(classification_report(y_test, ypred))
            plot_confusion(y_test, ypred)

            # compare to 92.6 threshold
            threshold = 0.926
            if best_acc >= threshold:
                st.success(f"🎉 Model beat the 92.6% benchmark (accuracy={best_acc:.4f})")
            else:
                st.info(f"Model did not beat 92.6% (best accuracy={best_acc:.4f}). You can try increasing GA generations/pop_size or use more data.")

# 3) Manual Prediction
elif nav == "Manual Prediction":
    st.header("Manual single-sample prediction")
    if st.session_state['dataset'] is None or st.session_state['trained_model'] is None:
        st.warning("Upload dataset and train a model first.")
    else:
        feat_cols = st.session_state['feature_columns']
        st.write("Enter values for features in order (or use the automatic form):")
        # Auto form using feature names
        with st.form("manual_form"):
            values = {}
            # for readability, show only up to 25 features by default; allow expand for more
            for c in feat_cols:
                # assume numeric after preprocessing
                values[c] = st.text_input(f"{c}", value="")
            submitted = st.form_submit_button("Predict")
        if submitted:
            try:
                arr = np.array([float(values[c]) for c in feat_cols]).reshape(1, -1)
                scaler = st.session_state['scaler']
                arr_s = scaler.transform(arr)
                model = st.session_state['trained_model']
                model_type = st.session_state['trained_model_type']
                if keras is not None and 'GA-ANN' in (model_type or "") and isinstance(model, keras.Model):
                    # detect multiclass
                    if model.output_shape[-1] > 1:
                        probs = model.predict(arr_s)
                        pred = int(np.argmax(probs, axis=1)[0])
                    else:
                        p = model.predict(arr_s).ravel()[0]
                        pred = int(p >= 0.5)
                else:
                    pred = int(model.predict(arr_s)[0])
                # decode if target label encoder exists
                target_le = st.session_state['label_encoder_target']
                if target_le is not None:
                    label = target_le.inverse_transform([pred])[0]
                else:
                    label = pred
                st.success(f"Prediction: {label}")
            except Exception as e:
                st.error("Prediction failed. Ensure you provided numeric values for all features in the same order. Error: " + str(e))

# 4) Bulk Prediction
elif nav == "Bulk Prediction":
    st.header("Bulk prediction (upload CSV of samples to predict)")
    if st.session_state['dataset'] is None or st.session_state['trained_model'] is None:
        st.warning("Upload dataset and train a model first.")
    else:
        st.write("Your training feature order will be used. Uploaded CSV must have same columns (names or order).")
        to_predict = st.file_uploader("Upload CSV for prediction (rows = samples)", type=["csv"], key="bulk_pred")
        if to_predict is not None:
            try:
                dfp = pd.read_csv(to_predict)
                st.write("Preview:")
                st.dataframe(dfp.head())
                # Align columns: try to re-order to session feature_columns if names match
                feat_cols = st.session_state['feature_columns']
                if set(feat_cols).issubset(set(dfp.columns)):
                    X_new = dfp[feat_cols].copy()
                else:
                    # If column names don't match, assume user provided same-order numeric array
                    X_new = dfp.iloc[:, :len(feat_cols)].copy()
                    X_new.columns = feat_cols
                # fill missing
                for c in X_new.columns:
                    if X_new[c].isnull().any():
                        X_new[c] = X_new[c].fillna(X_new[c].median())
                arr = X_new.values.astype(float)
                arr_s = st.session_state['scaler'].transform(arr)
                model = st.session_state['trained_model']
                model_type = st.session_state['trained_model_type']
                if keras is not None and 'GA-ANN' in (model_type or "") and isinstance(model, keras.Model):
                    if model.output_shape[-1] > 1:
                        probs = model.predict(arr_s)
                        preds = probs.argmax(axis=1)
                    else:
                        probs = model.predict(arr_s).ravel()
                        preds = (probs >= 0.5).astype(int)
                else:
                    preds = model.predict(arr_s)
                target_le = st.session_state['label_encoder_target']
                if target_le is not None:
                    labels = target_le.inverse_transform(preds.astype(int))
                else:
                    labels = preds
                out = dfp.copy()
                out['prediction'] = labels
                st.dataframe(out.head(50))
                st.download_button("Download predictions CSV", data=out.to_csv(index=False).encode('utf-8'), file_name='predictions.csv')
            except Exception as e:
                st.error("Bulk prediction failed: " + str(e))

# 5) Interpretability
elif nav == "Interpretability":
    st.header("Interpretability & Feature Importance")
    if st.session_state['dataset'] is None or st.session_state['trained_model'] is None:
        st.warning("Upload dataset and train a model first.")
    else:
        st.write("We will compute feature importances using:")
        st.write("- Tree feature_importances_ (if available)")
        st.write("- Permutation importance (model-agnostic)")
        st.write("- SHAP (if installed; may be slow)")

        model = st.session_state['trained_model']
        feat_cols = st.session_state['feature_columns']
        df_sample = st.session_state['dataset'].drop(columns=[st.session_state['target_column']])
        # ensure sample numeric + aligned
        try:
            # build X for importance (use up to 500 samples for speed)
            X_full = df_sample.copy()
            if set(feat_cols).issubset(set(X_full.columns)):
                X_full = X_full[feat_cols]
            else:
                X_full = X_full.iloc[:, :len(feat_cols)]
                X_full.columns = feat_cols
            for c in X_full.columns:
                if X_full[c].isnull().any():
                    X_full[c] = X_full[c].fillna(X_full[c].median())
            X_arr = X_full.values.astype(float)
            Xs = st.session_state['scaler'].transform(X_arr)
        except Exception as e:
            st.error("Could not prepare feature matrix for interpretability: " + str(e))
            Xs = None

        if Xs is not None:
            # tree-based
            if hasattr(model, "feature_importances_"):
                st.subheader("Model feature_importances_ (tree-based)")
                try:
                    fi = model.feature_importances_
                    imp_df = pd.DataFrame({"feature": feat_cols, "importance": fi}).sort_values("importance", ascending=False)
                    st.dataframe(imp_df)
                    fig, ax = plt.subplots(figsize=(6, min(0.4*len(imp_df), 6)))
                    sns.barplot(x="importance", y="feature", data=imp_df.head(30), ax=ax)
                    st.pyplot(fig)
                except Exception as e:
                    st.error("Failed to compute feature_importances_: " + str(e))
            else:
                st.info("Model is not tree-based or does not provide feature_importances_.")

            # permutation importance
            st.subheader("Permutation importance (model-agnostic)")
            try:
                # for permutation_importance we need a model that supports predict
                # create proxy y by using original labels
                y = st.session_state['dataset'][st.session_state['target_column']]
                if st.session_state['label_encoder_target'] is not None:
                    y_vals = st.session_state['label_encoder_target'].transform(y.astype(str))
                else:
                    y_vals = y.values
                # compute on subset for speed
                from sklearn.inspection import permutation_importance
                r = permutation_importance(model, Xs, y_vals, n_repeats=8, random_state=0, n_jobs=1)
                perm_df = pd.DataFrame({"feature": feat_cols, "importance": r.importances_mean}).sort_values("importance", ascending=False)
                st.dataframe(perm_df)
                fig, ax = plt.subplots(figsize=(6, min(0.4*len(perm_df), 6)))
                sns.barplot(x="importance", y="feature", data=perm_df.head(30), ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error("Permutation importance failed: " + str(e))

            # SHAP (optional)
            if shap is not None:
                st.subheader("SHAP explanations (sampled; may be slow)")
                try:
                    explainer = None
                    if 'XGB' in str(type(model)):
                        explainer = shap.TreeExplainer(model)
                        shap_vals = explainer.shap_values(Xs[:200])
                        st.pyplot(shap.summary_plot(shap_vals, Xs[:200], feature_names=feat_cols, show=False))
                    elif 'LGBM' in str(type(model)):
                        explainer = shap.TreeExplainer(model)
                        shap_vals = explainer.shap_values(Xs[:200])
                        st.pyplot(shap.summary_plot(shap_vals, Xs[:200], feature_names=feat_cols, show=False))
                    elif keras is not None and isinstance(model, keras.Model):
                        # Kernel explainer - sample small subset
                        sample_idx = np.random.choice(np.arange(Xs.shape[0]), size=min(100, Xs.shape[0]), replace=False)
                        explainer = shap.KernelExplainer(lambda x: model.predict(x).ravel(), Xs[sample_idx])
                        shap_vals = explainer.shap_values(Xs[sample_idx])
                        st.pyplot(shap.summary_plot(shap_vals, Xs[sample_idx], feature_names=feat_cols, show=False))
                    else:
                        st.info("SHAP explainer not configured for this model type.")
                except Exception as e:
                    st.error("SHAP failed or is too slow here: " + str(e))
            else:
                st.info("Install `shap` for richer explanations (optional).")

# End of app

