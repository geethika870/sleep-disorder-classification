# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sklearn
import warnings
warnings.filterwarnings("ignore")

# optional imports (if available)
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

# ANN + GA using tensorflow.keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    tf = None
    keras = None

# ---------- Streamlit app layout and helpers ----------
st.set_page_config(page_title="Sleep Disorder Classifier", layout="wide",
                   initial_sidebar_state="auto")

# CSS / Styling (colorful but no images)
st.markdown(
    """
    <style>
    .main {background: linear-gradient(90deg, #f0f8ff, #e6f7ff);}
    .stApp { background: linear-gradient(90deg, #f0f8ff, #e6f7ff); }
    .big-title {font-size:40px; font-weight:700; color:#0b486b;}
    .subtitle {font-size:16px; color:#0b486b;}
    .card {background: linear-gradient(180deg,#ffffff,#f7fdff); padding:16px; border-radius:12px; box-shadow: 0 6px 18px rgba(11,72,107,0.08);}
    </style>
    """, unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Upload & Explore", "Train & Evaluate", "Predict", "Interpretability"])

# ---------- Utility functions ----------
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

def summarize_df(df):
    st.write("Shape:", df.shape)
    st.dataframe(df.head(10))

def preprocess(df, target_col):
    df = df.copy()
    # Basic cleaning: drop rows with all NA
    df = df.dropna(how="all")
    # Simple imputation: numeric -> median; categorical -> mode
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # If target in categorical, encode later
    # Fill numeric
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "NA")
    # Label encode categorical columns
    encoders = {}
    for c in cat_cols:
        if c == target_col: 
            continue
        le = LabelEncoder()
        try:
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
        except Exception:
            pass
    # target
    y = df[target_col]
    if y.dtype == 'object' or y.dtype.name == 'category':
        y_enc = LabelEncoder().fit_transform(y.astype(str))
        target_mapping = dict(zip(LabelEncoder().fit(y.astype(str)).classes_, LabelEncoder().fit(y.astype(str)).transform(LabelEncoder().fit(y.astype(str)).classes_)))
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        target_mapping = None
    X = df.drop(columns=[target_col])
    return X, y, encoders, target_mapping

def plot_confusion(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    st.pyplot(fig)

# ---------- Genetic Algorithm for ANN (lightweight) ----------
def build_keras_model(input_dim, n_layers=1, n_neurons=16, activation='relu', lr=0.001, dropout=0.0):
    if keras is None:
        raise RuntimeError("TensorFlow/Keras is not installed.")
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    for _ in range(n_layers):
        model.add(layers.Dense(n_neurons, activation=activation))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def simple_ga_tune_ann(X_train, y_train, X_val, y_val, generations=6, pop_size=8, random_state=42):
    """
    Lightweight GA that searches these hyperparams:
     - n_layers: 1-3
     - n_neurons: 8-128
     - lr: 1e-4 - 1e-2 (log-uniform)
     - dropout: 0.0 - 0.4
     - activation: relu / tanh
    Returns best trained Keras model and its val accuracy.
    NOTE: This routine is intentionally conservative in epochs to run reasonably in a demo.
    """
    np.random.seed(random_state)
    param_space = {
        "n_layers": [1,2,3],
        "n_neurons": [8,16,32,64,128],
        "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
        "activation": ['relu', 'tanh']
    }

    def random_individual():
        return {
            "n_layers": int(np.random.choice(param_space['n_layers'])),
            "n_neurons": int(np.random.choice(param_space['n_neurons'])),
            "lr": float(np.random.choice(param_space['lr'])),
            "dropout": float(np.random.choice(param_space['dropout'])),
            "activation": str(np.random.choice(param_space['activation']))
        }

    def crossover(a, b):
        child = {}
        for k in a.keys():
            child[k] = a[k] if np.random.rand() < 0.5 else b[k]
        return child

    def mutate(ind, mut_rate=0.2):
        if np.random.rand() < mut_rate:
            key = np.random.choice(list(ind.keys()))
            ind[key] = random_individual()[key]
        return ind

    def fitness(ind):
        # train a small ANN and return val accuracy
        model = build_keras_model(input_dim=X_train.shape[1],
                                  n_layers=ind['n_layers'],
                                  n_neurons=ind['n_neurons'],
                                  activation=ind['activation'],
                                  lr=ind['lr'],
                                  dropout=ind['dropout'])
        # small number of epochs for speed; increase for better results
        history = model.fit(X_train, y_train, validation_data=(X_val,y_val),
                            epochs=25, batch_size=32, verbose=0)
        val_acc = history.history['val_accuracy'][-1]
        # to conserve memory, clear session
        keras.backend.clear_session()
        return val_acc

    # init population
    population = [random_individual() for _ in range(pop_size)]
    best = None
    for gen in range(generations):
        st.sidebar.write(f"GA generation {gen+1}/{generations}")
        # evaluate
        scored = []
        for ind in population:
            try:
                score = fitness(ind)
            except Exception as e:
                score = 0.0
            scored.append((score, ind))
        # sort
        scored.sort(key=lambda x: x[0], reverse=True)
        # keep best
        if best is None or scored[0][0] > best[0]:
            best = scored[0]
        # selection: top 50% survive
        survivors = [ind for (s, ind) in scored[: max(2, pop_size//2)]]
        # create new population via crossover+mutation
        new_pop = survivors.copy()
        while len(new_pop) < pop_size:
            a, b = np.random.choice(survivors, 2, replace=False)
            child = crossover(a, b)
            child = mutate(child, mut_rate=0.3)
            new_pop.append(child)
        population = new_pop
    # Build and return final best model trained on combined train+val
    best_params = best[1]
    final_model = build_keras_model(input_dim=(X_train.shape[1]),
                                    n_layers=best_params['n_layers'],
                                    n_neurons=best_params['n_neurons'],
                                    activation=best_params['activation'],
                                    lr=best_params['lr'],
                                    dropout=best_params['dropout'])
    final_model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]), epochs=40, batch_size=32, verbose=0)
    return final_model, best[0], best_params

# ---------- Pages ----------
if page == "Home":
    st.markdown('<div class="big-title">Sleep Disorder Classifier</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A demo app to classify sleep disorders and interpret model decisions.</p>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ### Why sleep matters
    - Sleep restores the brain and body, consolidates memory, and improves mood and immune function.
    - Sleep disorders (insomnia, sleep apnea, narcolepsy, restless legs, etc.) affect hundreds of millions worldwide.
    - Early screening using simple questionnaires and routine data can help triage high-risk individuals for further testing.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Reference (example related work):** A study that optimizes ANN with a Genetic Algorithm reports ~92.9% accuracy for multi-class sleep-disorder classification on a small benchmark dataset. This app aims to reproduce/beat such reported accuracy by trying multiple models and a GA-tuned ANN. :contentReference[oaicite:1]{index=1}")

elif page == "Upload & Explore":
    st.header("Upload Dataset (CSV)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = load_csv(uploaded)
        st.success("Loaded CSV")
        summarize_df(df)
        if st.checkbox("Show column types"):
            st.write(df.dtypes)
        if st.checkbox("Basic plots"):
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                col = st.selectbox("Choose numeric column to plot", num_cols)
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                st.pyplot(fig)
            else:
                st.info("No numeric columns found.")

elif page == "Train & Evaluate":
    st.header("Train & Evaluate Models")
    st.markdown("Upload dataset (same format as used in literature, e.g., Sleep Health & Lifestyle Dataset).")
    uploaded = st.file_uploader("CSV for training", type=["csv"], key="train")
    if uploaded is None:
        st.info("Upload a CSV to continue.")
    else:
        df = load_csv(uploaded)
        st.write("Dataset shape:", df.shape)
        all_cols = df.columns.tolist()
        target_col = st.selectbox("Select target column (label)", all_cols)
        test_size = st.slider("Test set proportion", 0.1, 0.4, 0.2)
        random_state = st.number_input("Random seed", value=42, step=1)
        run_auto = st.button("Run AUTO training (compare RF/XGB/LGBM/GA-ANN)")
        if run_auto:
            with st.spinner("Preprocessing..."):
                X, y, encs, mapping = preprocess(df, target_col)
                # Train/val split for GA
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y)
                # further split train into train+val for GA
                X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=int(random_state))
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_val_s = scaler.transform(X_val)
                X_test_s = scaler.transform(X_test)

            results = {}
            # RandomForest
            st.write("Training RandomForest...")
            rf = RandomForestClassifier(n_estimators=200, random_state=int(random_state))
            rf.fit(X_tr_s, y_tr)
            y_pred = rf.predict(X_test_s)
            rf_acc = accuracy_score(y_test, y_pred)
            results['RandomForest'] = (rf_acc, rf, scaler)
            st.write("RF accuracy:", rf_acc)

            # XGBoost
            if xgb is not None:
                st.write("Training XGBoost...")
                xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=int(random_state))
                xgb_clf.fit(X_tr_s, y_tr)
                y_pred = xgb_clf.predict(X_test_s)
                xgb_acc = accuracy_score(y_test, y_pred)
                results['XGBoost'] = (xgb_acc, xgb_clf, scaler)
                st.write("XGB accuracy:", xgb_acc)
            else:
                st.info("xgboost not installed — skipped.")

            # LightGBM
            if lgb is not None:
                st.write("Training LightGBM...")
                lgb_clf = lgb.LGBMClassifier(random_state=int(random_state))
                lgb_clf.fit(X_tr_s, y_tr)
                y_pred = lgb_clf.predict(X_test_s)
                lgb_acc = accuracy_score(y_test, y_pred)
                results['LightGBM'] = (lgb_acc, lgb_clf, scaler)
                st.write("LGB accuracy:", lgb_acc)
            else:
                st.info("lightgbm not installed — skipped.")

            # GA-tuned ANN (only if tensorflow installed)
            if keras is not None:
                st.write("Running GA tuner for ANN (this may take a while)...")
                try:
                    ann_model, best_val_acc, best_params = simple_ga_tune_ann(X_tr_s, y_tr, X_val_s, y_val, generations=4, pop_size=6)
                    # Evaluate on test
                    X_test_ann = X_test_s
                    y_test_pred_prob = ann_model.predict(X_test_ann).ravel()
                    y_test_pred = (y_test_pred_prob >= 0.5).astype(int)
                    ann_acc = accuracy_score(y_test, y_test_pred)
                    results['GA-ANN'] = (ann_acc, ann_model, scaler)
                    st.write("GA-ANN test accuracy:", ann_acc)
                    st.write("Best GA params:", best_params)
                except Exception as e:
                    st.error("GA-ANN failed: " + str(e))
            else:
                st.info("TensorFlow/Keras not installed — ANN skipped.")

            # pick best
            best_name = max(results.items(), key=lambda x: x[1][0])[0]
            best_acc, best_model, best_scaler = results[best_name]
            st.success(f"Best model: {best_name} with accuracy {best_acc:.4f}")
            # Save model
            joblib.dump({'model': best_model, 'scaler': best_scaler, 'type': type(best_model).__name__}, "best_model.joblib")
            st.write("Saved best_model.joblib to app folder.")

            # Show classification report and confusion for best model (handle keras)
            st.subheader("Evaluation — Best model")
            if best_name == 'GA-ANN' and keras is not None:
                y_prob = best_model.predict(best_scaler.transform(X_test))[:,0] if hasattr(best_model, "predict") else best_model.predict(X_test_s)
                y_pred = (y_prob >= 0.5).astype(int)
            else:
                y_pred = best_model.predict(best_scaler.transform(X_test))
            st.text(classification_report(y_test, y_pred))
            plot_confusion(y_test, y_pred)

            st.balloons()

elif page == "Predict":
    st.header("Predict new samples (manual or bulk CSV)")
    uploaded = st.file_uploader("Upload CSV with same features as training (no target column)", type=["csv"], key="predict")
    model_file = st.file_uploader("Upload trained model (best_model.joblib) or leave to use server model", type=["joblib", "pkl"], key="modelup")
    if 'best_model.joblib' in st.session_state:
        pass
    if model_file is None:
        # try to load saved model
        try:
            saved = joblib.load("best_model.joblib")
            model = saved['model']
            scaler = saved['scaler']
            st.success("Loaded server best_model.joblib")
        except Exception:
            model = None
            scaler = None
    else:
        saved = joblib.load(model_file)
        model = saved.get('model', None)
        scaler = saved.get('scaler', None)
        st.success("Uploaded model loaded.")

    if uploaded is not None and model is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview of input:")
        st.dataframe(df.head())
        # preprocess columns: assume already encoded / numeric if same schema as training
        try:
            Xp = df.values
            Xp_s = scaler.transform(Xp)
            if keras is not None and hasattr(model, "predict") and isinstance(model, keras.Model):
                probs = model.predict(Xp_s).ravel()
                preds = (probs >= 0.5).astype(int)
            else:
                preds = model.predict(Xp_s)
            out = df.copy()
            out['prediction'] = preds
            st.dataframe(out.head(20))
            st.download_button("Download predictions CSV", data=out.to_csv(index=False).encode('utf-8'), file_name='predictions.csv')
        except Exception as e:
            st.error("Prediction failed — ensure uploaded CSV has the same columns and numeric encoding as training. Error: " + str(e))

    st.markdown("---")
    st.subheader("Manual single sample prediction")
    if model is not None:
        # allow manual entry: build numeric fields from previously uploaded or let user paste CSV shape
        st.info("Manual input requires you to enter numeric features in the same order as training columns.")
        manual = st.text_area("Paste comma-separated numeric values for one sample (no target). Example: 23, 1, 7, 0, ...")
        if st.button("Predict manual sample"):
            if not manual:
                st.error("Provide a comma-separated line of numeric values.")
            else:
                try:
                    values = np.array([float(x.strip()) for x in manual.split(",")]).reshape(1, -1)
                    vals_s = scaler.transform(values)
                    if keras is not None and isinstance(model, keras.Model):
                        p = model.predict(vals_s).ravel()[0]
                        pred = int(p >= 0.5)
                    else:
                        pred = model.predict(vals_s)[0]
                    st.success(f"Prediction: {pred}")
                except Exception as e:
                    st.error("Failed to parse/ predict: " + str(e))
    else:
        st.info("No trained model available. Train a model first (Train & Evaluate).")

elif page == "Interpretability":
    st.header("Feature importance & interpretability")
    st.markdown("Upload your trained model (best_model.joblib) and a sample CSV used for training or testing to compute feature importance.")
    model_file = st.file_uploader("Upload trained model (.joblib)", type=["joblib", "pkl"], key="intp_model")
    data_file = st.file_uploader("Upload CSV with features (for permutation importance / SHAP)", type=["csv"], key="intp_data")
    if model_file and data_file:
        saved = joblib.load(model_file)
        model = saved.get('model', None)
        scaler = saved.get('scaler', None)
        df = pd.read_csv(data_file)
        X = df.values
        if scaler is not None:
            Xs = scaler.transform(X)
        else:
            Xs = X
        st.write("Using model:", type(model).__name__)
        # Tree-based importance
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            cols = df.columns.tolist()
            imp_df = pd.DataFrame({"feature": cols, "importance": fi}).sort_values("importance", ascending=False)
            st.dataframe(imp_df)
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x="importance", y="feature", data=imp_df.head(15), ax=ax)
            st.pyplot(fig)
        else:
            st.info("Model is not tree-based. Trying permutation importance...")
            try:
                # need a target y to compute permutation importance; ask user to upload with target?
                st.info("Permutation importance will estimate feature effect on model predictions (requires predictions only).")
                # create a fake y by predicting current model and using as proxy
                if keras is not None and isinstance(model, keras.Model):
                    y_proxy = (model.predict(Xs).ravel() >= 0.5).astype(int)
                else:
                    y_proxy = model.predict(Xs)
                r = permutation_importance(model, Xs, y_proxy, n_repeats=10, random_state=0)
                imp_df = pd.DataFrame({"feature": df.columns.tolist(), "importance": r.importances_mean}).sort_values("importance", ascending=False)
                st.dataframe(imp_df)
                fig, ax = plt.subplots(figsize=(6,4))
                sns.barplot(x="importance", y="feature", data=imp_df.head(15), ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.error("Permutation importance failed: " + str(e))

        # Try SHAP if available and model is supported
        if shap is not None:
            st.markdown("### SHAP explanations (if applicable)")
            try:
                explainer = None
                if 'XGB' in type(model).__name__ or 'XGB' in str(type(model)):
                    explainer = shap.TreeExplainer(model)
                elif 'LGBM' in type(model).__name__ or 'LightGBM' in str(type(model)):
                    explainer = shap.TreeExplainer(model)
                elif keras is not None and isinstance(model, keras.Model):
                    # KernelExplainer will be slow; show small subset
                    explainer = shap.KernelExplainer(lambda x: model.predict(x).ravel(), shap.sample(Xs, 50))
                else:
                    explainer = shap.KernelExplainer(lambda x: model.predict(x), shap.sample(Xs, 50))
                shap_vals = explainer.shap_values(shap.sample(Xs, min(100, Xs.shape[0])))
                st.success("Computed SHAP values (sampled). Visuals:")
                st.pyplot(shap.summary_plot(shap_vals, shap.sample(Xs, min(100, Xs.shape[0])), feature_names=df.columns))
            except Exception as e:
                st.error("SHAP explanation failed or is too slow in this environment: " + str(e))
        else:
            st.info("shap not installed — install shap for rich model explanations.")

# End of app
