import streamlit as st
import pandas as pd
import numpy as np
import pickle, joblib, os, random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTETomek

# ✅ Import DEAP correctly
from deap import base, creator, tools, algorithms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
st.sidebar.title("⚙ Navigation")
page = st.sidebar.radio("Go to:", ["📂 Upload Dataset", "🚀 Train Models", "🔮 Predict Disorder", "📊 Interpretability"])

# ✅ FIXED DEAP Genetic Algorithm Function
def ga_feature_selection(X_train, y_train):
    n_features = X_train.shape[1]

    # Reset creator to avoid duplicate creation errors
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    def eval_individual(ind):
        selected = [i for i in range(n_features) if ind[i] == 1]
        if not selected:
            return (0,)
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=SEED)
        clf.fit(X_train[:, selected], y_train)
        return (clf.score(X_train[:, selected], y_train),)

    toolbox = base.Toolbox()
    toolbox.register("attr", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=25)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=15, verbose=False)

    best = tools.selBest(pop, k=1)[0]
    return [i for i in range(n_features) if best[i] == 1]

def save_model(best_model, scaler, label_encoders, feature_order, selected_features=None):
    with open("best_model.pkl", "wb") as f:
        pickle.dump((best_model, scaler, label_encoders, feature_order, selected_features), f)

def load_model_file():
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            data = pickle.load(f)
            if len(data) == 4:
                return *data, None  # old save compatibility
            return data
    return None, None, None, None, None

# 📂 Upload Dataset
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
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(df.head())

# 🚀 Train Models
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare Models")
    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()

        # Encode categorical values
        label_encoders = {}
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        X = df.drop("Sleep Disorder", axis=1)
        y = df["Sleep Disorder"]

        # Balance dataset
        smt = SMOTETomek(random_state=SEED)
        X_res, y_res = smt.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, stratify=y_res, random_state=SEED
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ✅ Apply GA feature selection
        selected_features = ga_feature_selection(X_train_scaled, y_train)
        st.write("🧬 GA selected features:", selected_features)

        st.info("⏳ Training models...")

        models = {
            "SVM": SVC(C=1, kernel="rbf", probability=True, random_state=SEED),
            "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=20, random_state=SEED),
            "CatBoost": CatBoostClassifier(iterations=300, verbose=0, random_state=SEED),
            "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=SEED),
            "ANN+GA": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED)
        }

        # Train models and evaluate
        results = {}
        for name, model in models.items():
            if name == "ANN+GA":  # use GA selected features only
                model.fit(X_train_scaled[:, selected_features], y_train)
                results[name] = accuracy_score(y_test, model.predict(X_test_scaled[:, selected_features]))
            else:
                model.fit(X_train_scaled, y_train)
                results[name] = accuracy_score(y_test, model.predict(X_test_scaled))

        acc_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        acc_df["Accuracy"] = (acc_df["Accuracy"] * 100).round(2)
        st.table(acc_df)

        best_model_name = acc_df.iloc[acc_df["Accuracy"].idxmax()]["Model"]
        st.success(f"🏆 Best Model: {best_model_name}")

        st.session_state.best_model = models[best_model_name]
        st.session_state.scaler = scaler
        st.session_state.label_encoders = label_encoders
        st.session_state.feature_order = list(X.columns)
        st.session_state.selected_features = selected_features if best_model_name == "ANN+GA" else None

        # Confusion Matrix
        cm = confusion_matrix(y_test, st.session_state.best_model.predict(
            X_test_scaled[:, selected_features] if best_model_name == "ANN+GA" else X_test_scaled
        ))

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        if st.button("💾 Save Best Model"):
            save_model(models[best_model_name], scaler, label_encoders, list(X.columns), st.session_state.selected_features)
            st.success("✅ Model saved!")

# 🔮 Predict Disorder
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")
    best_model, scaler, label_encoders, feature_order, selected_features = load_model_file()

    if not best_model:
        st.warning("Train and save a model first!")
    else:
        st.session_state.best_model = best_model
        st.session_state.scaler = scaler
        st.session_state.label_encoders = label_encoders
        st.session_state.feature_order = feature_order
        st.session_state.selected_features = selected_features

        mode = st.radio("Prediction Mode", ["Manual Input", "Bulk Prediction"])

        if mode == "Manual Input":
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 1, 100, 25)
            occupation = st.selectbox("Occupation", ["Software Engineer", "Doctor", "Nurse", "Teacher", "Manager", "Student"])
            sleep_dur = st.slider("Sleep Duration", 3.0, 12.0, 7.0)
            q_sleep = st.slider("Quality of Sleep", 1, 10, 7)
            phys_act = st.slider("Physical Activity Level", 0, 100, 50)
            stress = st.slider("Stress Level", 1, 10, 5)
            bmi_cat = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese", "Underweight"])
            sys_bp = st.slider("Systolic BP", 80, 180, 120)
            dia_bp = st.slider("Diastolic BP", 50, 120, 80)
            hr = st.slider("Heart Rate", 40, 120, 70)
            steps = st.slider("Daily Steps", 0, 20000, 5000)

            user_data = pd.DataFrame([{
                "Gender": gender, "Age": age, "Occupation": occupation,
                "Sleep Duration": sleep_dur, "Quality of Sleep": q_sleep,
                "Physical Activity Level": phys_act, "Stress Level": stress,
                "BMI Category": bmi_cat, "Systolic_BP": sys_bp,
                "Diastolic_BP": dia_bp, "Heart Rate": hr, "Daily Steps": steps
            }])

            for col, le in label_encoders.items():
                if col in user_data.columns:
                    user_data[col] = user_data[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    user_data[col] = le.transform(user_data[col])

            user_data = user_data[feature_order]

            if st.button("🔮 Predict"):
                if selected_features:
                    scaled = scaler.transform(user_data)[:, selected_features]
                else:
                    scaled = scaler.transform(user_data)

                pred_num = best_model.predict(scaled)[0]
                target_encoder = label_encoders["Sleep Disorder"]
                pred_label = target_encoder.inverse_transform([pred_num])[0]

                st.success(f"🩺 Predicted Sleep Disorder: *{pred_label}*")

        else:
            file = st.file_uploader("Upload CSV without Sleep Disorder", type=["csv"])
            if file:
                new_df = pd.read_csv(file)

                if "Blood Pressure" in new_df.columns:
                    new_df[["Systolic_BP", "Diastolic_BP"]] = new_df["Blood Pressure"].str.split("/", expand=True).astype(int)
                    new_df.drop("Blood Pressure", axis=1, inplace=True)

                for col, le in label_encoders.items():
                    if col in new_df.columns:
                        new_df[col] = new_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        new_df[col] = le.transform(new_df[col])

                new_df = new_df[feature_order]
                if selected_features:
                    scaled = scaler.transform(new_df)[:, selected_features]
                else:
                    scaled = scaler.transform(new_df)

                preds = best_model.predict(scaled)
                preds_labels = label_encoders["Sleep Disorder"].inverse_transform(preds)
                new_df["Predicted_Sleep_Disorder"] = preds_labels
                st.dataframe(new_df.head())

# 📊 Interpretability
elif page == "📊 Interpretability":
    st.title("📊 Model Interpretability - Feature Importance")
    best_model, scaler, label_encoders, feature_order, selected_features = load_model_file()

    if not best_model:
        st.warning("No saved model found! Train first.")
    else:
        X = st.session_state.df[feature_order]
        y = st.session_state.df["Sleep Disorder"]

        y_encoded = label_encoders["Sleep Disorder"].transform(y)
        X_scaled = scaler.transform(X)

        if selected_features:
            X_scaled = X_scaled[:, selected_features]

        st.info("⏳ Calculating permutation importance...")
        result = permutation_importance(
            best_model, X_scaled, y_encoded,
            n_repeats=10, random_state=SEED, scoring="accuracy"
        )

        sorted_idx = result.importances_mean.argsort()[::-1]
        features = np.array(feature_order if not selected_features else np.array(feature_order)[selected_features])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=result.importances_mean[sorted_idx], y=features[sorted_idx], ax=ax)
        ax.set_title("Permutation Feature Importance")
        ax.set_xlabel("Mean Importance")
        st.pyplot(fig)
        st.success("✅ Feature importance calculated!")

