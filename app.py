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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
st.sidebar.title("⚙ Navigation")
page = st.sidebar.radio("Go to:", ["📂 Upload Dataset", "🚀 Train Models", "🔮 Predict Disorder", "📊 Interpretability"])

# ✅ FEATURE ORDER SAVE & LOAD
def save_model(best_model, scaler, label_encoders, feature_order):
    with open("best_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler, "encoders": label_encoders, "features": feature_order}, f)

def load_model_file():
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            data = pickle.load(f)
            return data["model"], data["scaler"], data["encoders"], data["features"]
    return None, None, None, None

# ✅ OPTIONAL GA IMPORT WITH CLOUD FALLBACK
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ModuleNotFoundError:
    DEAP_AVAILABLE = False

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

# 📂 Upload Dataset
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        # clean columns
        if "Person ID" in df.columns:
            df.drop("Person ID", axis=1, inplace=True)
        if "Blood Pressure" in df.columns:
            df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
            df.drop("Blood Pressure", axis=1, inplace=True)

        df.dropna(inplace=True)  # remove bad rows
        st.session_state.df = df
        st.success("✅ Dataset uploaded!")
        st.dataframe(df.head())

# 🚀 TRAIN MODELS
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare Models")
    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()

        # encode all object columns
        label_encoders = {}
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        if "Sleep Disorder" in df.columns:
            X = df.drop("Sleep Disorder", axis=1)
            y = df["Sleep Disorder"]
        else:
            st.error("Dataset must contain 'Sleep Disorder'")
            st.stop()

        # balance dataset
        smt = SMOTETomek(random_state=SEED)
        X, y = smt.fit_resample(X, y)

        # drop NaN and ensure numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        X.fillna(X.mean(), inplace=True)

        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

        # scale once
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # optional GA feature selection fallback
        if DEAP_AVAILABLE:
            def ga_feature_selection(X_train, y_train):
                n_features = X_train.shape[1]
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMax)

                def eval_individual(ind):
                    selected = [i for i in range(n_features) if ind[i] == 1]
                    if not selected:
                        return (0,)
                    ann = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=150, random_state=SEED)
                    ann.fit(X_train[:, selected], y_train)
                    return (ann.score(X_train[:, selected], y_train),)

                toolbox = base.Toolbox()
                toolbox.register("attr", random.randint, 0, 1)
                toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, n=n_features)
                toolbox.register("population", tools.initRepeat, list, toolbox.individual)
                toolbox.register("evaluate", eval_individual)
                toolbox.register("mate", tools.cxTwoPoint)
                toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
                toolbox.register("select", tools.selTournament, tournsize=3)
                pop = toolbox.population(n=15)
                algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=8, verbose=False)
                best = tools.selBest(pop, 1)[0]
                return [i for i in range(n_features) if best[i] == 1]

            selected = ga_feature_selection(X_train, y_train)
        else:
            selected = list(range(X_train.shape[1]))
            st.warning("⚠ GA skipped (DEAP unavailable), training on all features.")

        # fast models only
        fast_models = {
            "SVM": SVC(kernel="linear", probability=True, random_state=SEED),
            "RF": RandomForestClassifier(n_estimators=120, max_depth=12, random_state=SEED),
            "LightGBM": LGBMWrapper(LGBMClassifier(n_estimators=150, learning_rate=0.07, random_state=SEED)),
            "CatBoost": CatBoostClassifier(iterations=150, depth=5, verbose=0, random_state=SEED),
            "XGBoost": XGBClassifier(n_estimators=140, max_depth=6, eval_metric="mlogloss", random_state=SEED),
            "ANN+GA": MLPClassifier(hidden_layer_sizes=(32,16), max_iter=200, random_state=SEED)
        }

        # 🧠 CACHE TRAINING SO IT DOES NOT RUN AGAIN AND AGAIN
        @st.cache_resource
        def train_all():
            trained = {}
            scores = {}
            for name, model in fast_models.items():
                if name == "ANN+GA":
                    model.fit(X_train[:, selected], y_train)
                    y_pred = model.predict(X_test[:, selected])
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred) * 100
                trained[name] = (model, acc)
                scores[name] = round(acc, 2)

            return trained, scores

        trained, results = train_all()

        # accuracy table
        acc_df = pd.DataFrame({"Model": results.keys(), "Accuracy": results.values()})
        st.table(acc_df)

        best_model_name = acc_df.iloc[acc_df["Accuracy"].idxmax()]["Model"]
        st.success(f"🏆 Best Model: {best_model_name}")

        best_model, _ = trained[best_model_name]

        st.session_state.best_model = best_model
        st.session_state.scaler = scaler
        st.session_state.label_encoders = label_encoders
        st.session_state.feature_order = list(X.columns)
        st.session_state.selected = selected

        if st.button("💾 Save Best Model"):
            save_model(best_model, scaler, label_encoders, st.session_state.feature_order)
            st.success("✅ Saved!")

# 🔮 Prediction
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")
    model, scaler, enc, features = load_model_file()

    if model is None:
        st.warning("Train or upload a saved model!")
    else:
        st.session_state.best_model = model
        st.session_state.scaler = scaler
        st.session_state.label_encoders = enc
        st.session_state.feature_order = features
        st.session_state.selected = enc.get("selected", [])

        st.success("✅ Model Loaded!")

        disorder = st.text_input("Enter Disorder")
        if disorder and st.button("🔮 Predict"):
            X = pd.DataFrame([[random.random() for _ in features]], columns=features)
            X = X.apply(pd.to_numeric, errors='coerce')
            X.fillna(X.mean(), inplace=True)
            X = scaler.transform(X)

            if DEAP_AVAILABLE:
                pred = model.predict(X[:, st.session_state.selected])[0]
            else:
                pred = model.predict(X)[0]

            st.write("Prediction:", disorder, "->", pred)

# 📊 Interpretability
elif page == "📊 Interpretability":
    st.title("📊 Interpretability")
    if "best_model" not in st.session_state:
        st.warning("Train or load model!")
    else:
        df = st.session_state.df.copy()
        X = df[st.session_state.feature_order]

        X = X.apply(pd.to_numeric, errors='coerce')
        X.fillna(X.mean(), inplace=True)

        try:
            X_scaled = st.session_state.scaler.transform(X)
        except Exception:
            st.error("⚠ Scaling failed — Dataset mismatch. Make sure same columns used.")
            st.stop()

        result = permutation_importance(st.session_state.best_model, X_scaled, df["Sleep Disorder"], n_repeats=5, random_state=SEED)

        imp_df = pd.DataFrame({
            "Feature": st.session_state.feature_order,
            "Importance": result.importances_mean
        }).sort_values(by="Importance", ascending=False)

        st.table(imp_df)

