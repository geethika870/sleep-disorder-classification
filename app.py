import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, random, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTETomek

# ==============================
# FIXED SEED
# ==============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction (GA+ANN)", layout="wide")
st.sidebar.title("⚙ Navigation")

page = st.sidebar.radio("Go to:", [
    "📂 Upload Dataset",
    "🚀 Train Model (GA+ANN)",
    "🔮 Predict Disorder"
])

# ==========================================================
# SAVE / LOAD MODEL
# ==========================================================
def save_model(best_model, scaler, encoders, order):
    with open("best_ga_model.pkl", "wb") as f:
        pickle.dump((best_model, scaler, encoders, order), f)

def load_model():
    if os.path.exists("best_ga_model.pkl"):
        with open("best_ga_model.pkl", "rb") as f:
            return pickle.load(f)
    return None, None, None, None

# ==========================================================
# GENETIC ALGORITHM FOR ANN OPTIMIZATION
# ==========================================================
def random_individual():
    return {
        "layer1": random.choice([32, 64, 128, 256]),
        "layer2": random.choice([0, 32, 64, 128]),
        "activation": random.choice(["relu", "tanh"]),
        "lr": random.choice([0.0005, 0.001, 0.005]),
        "batch": random.choice([8, 16, 32])
    }

def crossover(parent1, parent2):
    child = {}
    for k in parent1.keys():
        child[k] = parent1[k] if random.random() < 0.5 else parent2[k]
    return child

def mutate(ind):
    key = random.choice(list(ind.keys()))
    if key == "layer1":
        ind[key] = random.choice([32, 64, 128, 256])
    elif key == "layer2":
        ind[key] = random.choice([0, 32, 64, 128])
    elif key == "activation":
        ind[key] = random.choice(["relu", "tanh"])
    elif key == "lr":
        ind[key] = random.choice([0.0005, 0.001, 0.005])
    elif key == "batch":
        ind[key] = random.choice([8, 16, 32])
    return ind

def evaluate(individual, X_train, X_test, y_train, y_test):
    layers = tuple([x for x in [individual["layer1"], individual["layer2"]] if x != 0])

    model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation=individual["activation"],
        learning_rate_init=individual["lr"],
        batch_size=individual["batch"],
        max_iter=500,
        random_state=SEED
    )

    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return acc, model
    except:
        return 0, None

# ==========================================================
# UPLOAD PAGE
# ==========================================================
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "Person ID" in df.columns:
            df.drop("Person ID", axis=1, inplace=True)

        if "Blood Pressure" in df.columns:
            parts = df["Blood Pressure"].str.split("/", expand=True)
            df["Systolic_BP"] = parts[0].astype(int)
            df["Diastolic_BP"] = parts[1].astype(int)
            df.drop("Blood Pressure", axis=1, inplace=True)

        st.session_state.df = df
        st.success("Uploaded Successfully")
        st.dataframe(df.head())

# ==========================================================
# TRAIN PAGE (GA + ANN)
# ==========================================================
elif page == "🚀 Train Model (GA+ANN)":
    st.title("🚀 Train ANN with Genetic Algorithm Optimization")

    if "df" not in st.session_state:
        st.warning("Upload dataset first.")
    else:
        df = st.session_state.df.copy()

        if "Sleep Disorder" not in df.columns:
            st.error("Target column 'Sleep Disorder' missing.")
        else:
            encoders = {}
            for col in df.select_dtypes(include="object").columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

            X = df.drop("Sleep Disorder", axis=1)
            y = df["Sleep Disorder"]

            # Balance classes
            sm = SMOTETomek(random_state=SEED)
            X, y = sm.fit_resample(X, y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            st.info("⏳ Running Genetic Algorithm... (10 individuals × 5 generations)")

            population = [random_individual() for _ in range(10)]
            best_score = 0
            best_model = None
            best_individual = None

            progress = st.progress(0)

            for gen in range(5):
                scores = []
                for ind in population:
                    acc, model = evaluate(ind, X_train, X_test, y_train, y_test)
                    scores.append((acc, ind, model))

                scores.sort(reverse=True, key=lambda x: x[0])

                if scores[0][0] > best_score:
                    best_score, best_individual, best_model = scores[0]

                new_pop = [scores[0][1], scores[1][1]]  # elitism

                while len(new_pop) < 10:
                    p1, p2 = random.sample(scores[:5], 2)
                    child = crossover(p1[1], p2[1])
                    if random.random() < 0.3:
                        child = mutate(child)
                    new_pop.append(child)

                population = new_pop
                progress.progress((gen + 1) / 5)

            st.success(f"🏆 Best GA Accuracy: {round(best_score * 100, 2)}%")
            st.write("Best ANN Architecture:")
            st.json(best_individual)

            # Save model
            save_model(best_model, scaler, encoders, list(X.columns))
            st.success("Model Saved ✔")

# ==========================================================
# PREDICT PAGE
# ==========================================================
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")

    model, scaler, encoders, order = load_model()

    if model is None:
        st.warning("Train the model first.")
    else:
        mode = st.radio("Mode:", ["Manual Input", "Upload CSV"])

        if mode == "Manual Input":
            inputs = {}
            for col in order:
                if col.lower() in ["gender"]:
                    inputs[col] = st.selectbox(col, ["Male", "Female"])
                else:
                    inputs[col] = st.number_input(col, value=0.0)

            if st.button("Predict"):
                df = pd.DataFrame([inputs])

                for col, le in encoders.items():
                    df[col] = le.transform(df[col].astype(str))

                df = df[order]
                scaled = scaler.transform(df)
                pred = model.predict(scaled)[0]

                target_le = encoders["Sleep Disorder"]
                label = target_le.inverse_transform([pred])[0]

                st.success(f"Predicted: {label}")

        else:
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                new_df = pd.read_csv(file)

                for col, le in encoders.items():
                    new_df[col] = le.transform(new_df[col].astype(str))

                new_df = new_df[order]
                scaled = scaler.transform(new_df)
                preds = model.predict(scaled)

                target_le = encoders["Sleep Disorder"]
                labels = target_le.inverse_transform(preds)

                new_df["Predicted_Sleep_Disorder"] = labels
                st.dataframe(new_df)
