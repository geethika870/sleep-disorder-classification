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
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
st.sidebar.title("⚙ Navigation")
page = st.sidebar.radio("Go to:", ["📂 Upload Dataset", "🚀 Train Models", "🔮 Predict Disorder", "📊 Interpretability"])

# ---- GA fallback (cloud safe) ----
try:
    from deap import base, creator, tools, algorithms
    GA_AVAILABLE = True
except ModuleNotFoundError:
    GA_AVAILABLE = False

# ---- Model Save / Load ----
def save_model(best_model, scaler, encoders, features, selected=None):
    data = {
        "model": best_model,
        "scaler": scaler,
        "encoders": encoders,
        "features": features,
        "selected": selected or []
    }
    with open("best_model.pkl", "wb") as f:
        pickle.dump(data, f)

def load_model_file():
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            d = pickle.load(f)
            return d["model"], d["scaler"], d["encoders"], d["features"], d.get("selected", [])
    return None, None, None, None, None

class LGBMWrapper:
    def __init__(self, model):
        self.model = model
    def fit(self, X, y): return self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)
    def predict_proba(self, X): return self.model.predict_proba(X)
    def __getattr__(self, a): return getattr(self.model, a)

# ---- Upload Page ----
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        if "Person ID" in df.columns:
            df.drop("Person ID", axis=1, inplace=True)

        if "Blood Pressure" in df.columns:
            df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True)
            df["Systolic_BP"] = pd.to_numeric(df["Systolic_BP"], errors='coerce')
            df["Diastolic_BP"] = pd.to_numeric(df["Diastolic_BP"], errors='coerce')
            df.drop("Blood Pressure", axis=1, inplace=True)

        df.dropna(inplace=True)
        st.session_state.df = df
        st.success("✅ Dataset uploaded!")
        st.dataframe(df.head())

# ---- Train Page ----
elif page == "🚀 Train Models":
    st.title("🚀 Train & Compare Models")

    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
        st.stop()

    df = st.session_state.df.copy()

    # encode categoricals
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    if "Sleep Disorder" not in df.columns:
        st.error("Target column 'Sleep Disorder' missing!")
        st.stop()

    X = df.drop("Sleep Disorder", axis=1)
    y = df["Sleep Disorder"]

    # ensure numeric & finite
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(X.mean(), inplace=True)

    # balance
    smt = SMOTETomek(random_state=SEED)
    X_res, y_res = smt.fit_resample(X, y)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=SEED
    )

    # scale ONCE
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # GA-based feature select (only if available)
    if GA_AVAILABLE:
        n_feat = X_train.shape[1]

        try: del creator.FitnessMax, creator.Individual
        except: pass

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        def eval_ind(ind):
            sel = [i for i in range(n_feat) if ind[i] == 1]
            if not sel: return (0,)
            ann = MLPClassifier((32,16), max_iter=120, random_state=SEED)
            ann.fit(X_train[:, sel], y_train)
            return (ann.score(X_train[:, sel], y_train),)

        tb = base.Toolbox()
        tb.register("attr", random.randint, 0, 1)
        tb.register("ind", tools.initRepeat, creator.Individual, tb.attr, n=n_feat)
        tb.register("pop", tools.initRepeat, list, tb.ind)
        tb.register("evaluate", eval_ind)
        tb.register("mate", tools.cxTwoPoint)
        tb.register("mutate", tools.mutFlipBit, indpb=0.05)
        tb.register("select", tools.selTournament, tournsize=3)

        pop = tb.pop(n=14)
        algorithms.eaSimple(pop, tb, cxpb=0.5, mutpb=0.2, ngen=6, verbose=False)
        best = tools.selBest(pop, 1)[0]
        selected = [i for i in range(n_feat) if best[i] == 1]
        st.info(f"🧬 GA selected {len(selected)} features")

    else:
        selected = list(range(X_train.shape[1]))
        st.warning("⚠ GA not installed, training on all features")

    # models (lightweight for speed)
    fast_models = {
        "SVM": SVC(kernel="linear", probability=True, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=130, max_depth=12, random_state=SEED),
        "LightGBM": LGBMWrapper(LGBMClassifier(n_estimators=150, learning_rate=0.07, random_state=SEED)),
        "XGBoost": XGBClassifier(n_estimators=140, max_depth=6, eval_metric="mlogloss", random_state=SEED),
        "CatBoost": CatBoostClassifier(iterations=160, depth=5, verbose=0, random_state=SEED),
        "ANN+GA": MLPClassifier((32,16), max_iter=250, random_state=SEED)
    }

    # ---- cache and train ----
    @st.cache_resource
    def train():
        scores = {}
        trained = {}
        for name, model in fast_models.items():

            if name == "ANN+GA" and selected:
                model.fit(X_train[:, selected], y_train)
                y_pred = model.predict(X_test[:, selected])
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            scores[name] = round(accuracy_score(y_test, y_pred) * 100, 2)
            trained[name] = model

        return scores, trained

    scores, trained_models = train()

    # display accuracies
    acc_df = pd.DataFrame({"Model": scores.keys(), "Accuracy(%)": scores.values()})
    st.table(acc_df)

    best_name = acc_df.iloc[acc_df["Accuracy(%)"].idxmax()]["Model"]
    best_model = trained_models[trained_models.keys().__iter__().__next__()] if best_name not in trained_models else trained_models[best_name]

    st.session_state.best_model = fast_models[best_name]
    st.session_state.scaler = scaler
    st.session_state.label_encoders = encoders
    st.session_state.feature_order = list(X.columns)
    st.session_state.selected = selected

    # confusion matrix
    cm = confusion_matrix(y_test, fast_models[best_name].predict(X_test[:, selected] if best_name=="ANN+GA" else X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    if st.button("💾 Save Best Model"):
        save_model(fast_models[best_name], scaler, encoders, st.session_state.feature_order, selected)
        st.success("✅ Saved to best_model.pkl")

# ---- Prediction Page ----
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")

    model, scaler, encoders, features, selected = load_model_file()
    if model is None:
        st.warning("Train models first or upload a saved model!")
        st.stop()

    st.session_state.best_model = model
    st.session_state.scaler = scaler
    st.session_state.label_encoders = encoders
    st.session_state.feature_order = features
    st.session_state.selected = selected or []

    mode = st.radio("Prediction Mode", ["🧍 Manual", "📑 Bulk"])

    if mode == "🧍 Manual":
        inputs = {}
        for f in features:
            inputs[f] = st.number_input(f, value=0.0)

        Xp = pd.DataFrame([inputs])
        Xp = Xp.apply(pd.to_numeric, errors='coerce').fillna(Xp.mean())
        Xp = scaler.transform(Xp)

        if st.button("🔮 Predict"):
            if st.session_state.selected and GA_AVAILABLE:
                pred = model.predict(Xp[:, st.session_state.selected])[0]
            else:
                pred = model.predict(Xp)[0]

            out = encoders["Sleep Disorder"].inverse_transform([pred])[0]
            st.success(f"🩺 **Predicted Sleep Disorder: {out}**")

    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            dfp = pd.read_csv(file)
            # create BP columns if needed
            if "Blood Pressure" in dfp.columns:
                dfp[["Systolic_BP","Diastolic_BP"]] = dfp["Blood Pressure"].str.split("/",expand=True)
                dfp.drop("Blood Pressure",axis=1,inplace=True)

            dfp = dfp[features]
            dfp = dfp.apply(pd.to_numeric, errors='coerce').fillna(dfp.mean())
            dfp = scaler.transform(dfp)

            if st.session_state.selected and GA_AVAILABLE:
                preds = model.predict(dfp[:, st.session_state.selected])
            else:
                preds = model.predict(dfp)

            dfp = pd.DataFrame(scaler.inverse_transform(dfp), columns=features)
            dfp["Predicted Sleep Disorder"] = encoders["Sleep Disorder"].inverse_transform(preds)
            st.dataframe(dfp.head())

# ---- Interpretability Page ----
elif page == "📊 Interpretability":
    st.title("📊 Feature Importance")

    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
        st.stop()

    df = st.session_state.df.copy()
    X = df[st.session_state.feature_order]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())

    try:
        X_scaled = st.session_state.scaler.transform(X)
    except:
        st.error("⚠ Feature mismatch — retrain model with same dataset")
        st.stop()

    res = permutation_importance(st.session_state.best_model, X_scaled, df["Sleep Disorder"], n_repeats=5, random_state=SEED)
    imp = pd.DataFrame({
        "Feature": st.session_state.feature_order,
        "Importance": res.importances_mean
    }).sort_values("Importance",ascending=False)

    st.table(imp)

