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


def save_model(best_model, scaler, label_encoders, feature_order, selected_features=None):
    # Save real model objects, remove any wrappers
    with open("best_model.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "scaler": scaler,
            "encoders": label_encoders,
            "features": feature_order,
            "ga_feats": selected_features
        }, f)


def load_model_file():
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            data = pickle.load(f)
            return data["model"], data["scaler"], data["encoders"], data["features"], data.get("ga_feats")
    return None, None, None, None, None


# ---------- UPLOAD ----------
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if "Person ID" in df.columns:
            df.drop("Person ID", axis=1, inplace=True)
        if "Blood Pressure" in df.columns:
            df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
            df.drop("Blood Pressure", axis=1, inplace=True)

        st.session_state.df = df
        st.success("✅ Dataset uploaded!")
        st.dataframe(df.head())


# ---------- TRAIN ----------
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare Models")

    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()

        # Encode categoricals
        label_encoders = {}
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Split X, y
        X = df.drop("Sleep Disorder", axis=1)
        y = df["Sleep Disorder"]

        # Resample
        smt = SMOTETomek(random_state=SEED)
        X_res, y_res = smt.fit_resample(X, y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, stratify=y_res, random_state=SEED
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ---- GA Feature selection for ANN ----
        def ga_feature_selection(X_train, y_train):
            n_features = X_train.shape[1]

            # reset GA creators if rerun
            try:
                del creator.FitnessMax, creator.Individual
            except:
                pass

            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

            def eval_ind(ind):
                selected = [i for i, bit in enumerate(ind) if bit == 1]
                if len(selected) == 0:
                    return (0,)
                clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=SEED)
                clf.fit(X_train[:, selected], y_train)
                return (clf.score(X_train[:, selected], y_train),)

            toolbox = base.Toolbox()
            toolbox.register("attr_bool", random.randint, 0, 1)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", eval_ind)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # ✅ FIXED
            toolbox.register("select", tools.selTournament, tournsize=3)

            pop = toolbox.population(n=20)
            algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=15, verbose=False)

            best = tools.selBest(pop, 1)[0]
            return [i for i, bit in enumerate(best) if bit == 1]

        selected_features = ga_feature_selection(X_train_scaled, y_train)
        st.write("🧬 GA Selected Feature Indexes:", selected_features)

        # Train base models
        models = {
            "SVM": SVC(C=1, kernel="rbf", probability=True),
            "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=20, random_state=SEED),
            "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED),
            "CatBoost": CatBoostClassifier(iterations=300, verbose=0, random_state=SEED),
            "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=SEED),
            "ANN": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED),
            "ANN+GA": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED)
        }

        # Fit models (ANN+GA gets selected columns only)
        for name, model in models.items():
            if name == "ANN+GA":
                model.fit(X_train_scaled[:, selected_features], y_train)
            else:
                model.fit(X_train_scaled, y_train)

        # Evaluate
        results = {}
        for name, model in models.items():
            if name == "ANN+GA":
                y_pred = model.predict(X_test_scaled[:, selected_features])
            else:
                y_pred = model.predict(X_test_scaled)
            results[name] = accuracy_score(y_test, y_pred)

        acc_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
        acc_df["Accuracy"] = (acc_df["Accuracy"] * 100).round(2)
        acc_df = acc_df.reset_index().rename(columns={"index": "Model"})
        st.table(acc_df)

        # Pick winner
        best_model_name = acc_df.iloc[acc_df["Accuracy"].idxmax()]["Model"]
        st.success(f"🏆 Best Model: {best_model_name}")

        # Save correct model, scaler, encoders
        st.session_state.best_model = models[best_model_name]
        st.session_state.scaler = scaler
        st.session_state.label_encoders = label_encoders
        st.session_state.feature_order = list(X.columns)
        st.session_state.selected_features = selected_features if best_model_name == "ANN+GA" else None

        # Confusion Matrix
        if best_model_name == "ANN+GA":
            cm = confusion_matrix(y_test, models["ANN+GA"].predict(X_test_scaled[:, selected_features]))
        else:
            cm = confusion_matrix(y_test, models[best_model_name].predict(X_test_scaled))

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        if st.button("💾 Save Best Model"):
            selected = selected_features if best_model_name == "ANN+GA" else None
            save_model(models[best_model_name], scaler, label_encoders, list(X.columns), selected)
            st.success("✅ Model saved safely!")


# ---------- PREDICT ----------
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")

    model, scaler, encoders, features, ga_feats = load_model_file()

    # Auto-load to session if saved
    if model:
        st.session_state.best_model = model
        st.session_state.scaler = scaler
        st.session_state.label_encoders = encoders
        st.session_state.feature_order = features
        st.session_state.selected_features = ga_feats

    if "best_model" not in st.session_state:
        st.warning("Train or load a model first!")
    else:
        pred_mode = st.radio("Mode:", ["Manual", "Bulk"])

        if pred_mode == "Manual":
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 1, 100, 25)
            occ = st.selectbox("Occupation", ["Software Engineer", "Doctor", "Nurse", "Teacher", "Manager", "Student"])
            sleep = st.slider("Sleep Duration", 3.0, 12.0, 7.0)
            quality = st.slider("Sleep Quality", 1, 10, 7)
            pa = st.slider("Physical Activity", 0, 100, 50)
            stress = st.slider("Stress", 1, 10, 5)
            bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese", "Underweight"])
            sbp = st.slider("Systolic", 80, 180, 120)
            dbp = st.slider("Diastolic", 50, 120, 80)
            hr = st.slider("Heart Rate", 40, 120, 70)
            steps = st.slider("Daily Steps", 0, 20000, 5000)

            input_df = pd.DataFrame([{
                "Gender": gender, "Age": age, "Occupation": occ,
                "Sleep Duration": sleep, "Quality of Sleep": quality,
                "Physical Activity Level": pa, "Stress Level": stress,
                "BMI Category": bmi, "Systolic_BP": sbp,
                "Diastolic_BP": dbp, "Heart Rate": hr, "Daily Steps": steps
            }])

            for c, le in st.session_state.label_encoders.items():
                if c in input_df.columns:
                    if input_df[c].iloc[0] not in le.classes_:
                        le.classes_ = np.append(le.classes_, input_df[c].iloc[0])
                    input_df[c] = le.transform(input_df[c])

            input_df = input_df[st.session_state.feature_order]
            Xs = st.session_state.scaler.transform(input_df)

            if st.session_state.selected_features is not None:
                Xs = Xs[:, st.session_state.selected_features]

            if st.button("Predict"):
                num = st.session_state.best_model.predict(Xs)[0]
                label = st.session_state.label_encoders["Sleep Disorder"].inverse_transform([num])[0]
                st.success(f"🩺 Result: {label}")

        else:
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                bulk_df = pd.read_csv(file)

                if "Blood Pressure" in bulk_df.columns:
                    bulk_df[["Systolic_BP", "Diastolic_BP"]] = bulk_df["Blood Pressure"].str.split("/", expand=True).astype(int)
                    bulk_df.drop("Blood Pressure", axis=1, inplace=True)

                for c, le in st.session_state.label_encoders.items():
                    if c in bulk_df.columns:
                        bulk_df[c] = bulk_df[c].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        bulk_df[c] = le.transform(bulk_df[c])

                bulk_df = bulk_df[st.session_state.feature_order]
                Xs = scaler.transform(bulk_df)

                if st.session_state.selected_features is not None:
                    Xs = Xs[:, st.session_state.selected_features]

                ps = st.session_state.best_model.predict(Xs)
                bulk_df["Predicted_Sleep_Disorder"] = encoders["Sleep Disorder"].inverse_transform(ps)
                st.dataframe(bulk_df.head())


# ---------- INTERPRET ----------
elif page == "📊 Interpretability":
    st.title("📊 Model Interpretability")

    if "df" not in st.session_state or "best_model" not in st.session_state:
        st.warning("Upload + train first!")
    else:
        df = st.session_state.df.copy()
        enc = st.session_state.label_encoders
        features = st.session_state.feature_order
        scaler = st.session_state.scaler
        model = st.session_state.best_model

        for c, le in enc.items():
            if c in df.columns and df[c].dtype == object:
                df[c] = df[c].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[c] = le.transform(df[c])

        X = df[features]
        y = enc["Sleep Disorder"].transform(df["Sleep Disorder"]) if df["Sleep Disorder"].dtype == object else df["Sleep Disorder"]
        Xs = scaler.transform(X)
        if st.session_state.selected_features is not None:
            Xs = Xs[:, st.session_state.selected_features]

        imp = permutation_importance(model, Xs, y, n_repeats=10, random_state=SEED)
        idx = imp.importances_mean.argsort()[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=imp.importances_mean[idx], y=np.array(features)[idx], ax=ax)
        st.pyplot(fig)
        st.success("✅ Importance computed!")

