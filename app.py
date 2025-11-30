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
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
st.sidebar.title("⚙ Navigation")
page = st.sidebar.radio("Go to:", ["📂 Upload Dataset", "🚀 Train Models", "🔮 Predict Disorder", "📊 Interpretability"])

def save_model(model, scaler, encoders, order):
    with open("best_model.pkl", "wb") as f:
        pickle.dump((model, scaler, encoders, order), f)

def load_model():
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            return pickle.load(f)
    return None, None, None, None

# -------- UPLOAD DATASET --------
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        if "Person ID" in df.columns:
            df.drop("Person ID", axis=1, inplace=True)

        if "Blood Pressure" in df.columns:
            df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True)
            df["Systolic_BP"] = df["Systolic_BP"].astype(int)
            df["Diastolic_BP"] = df["Diastolic_BP"].astype(int)
            df.drop("Blood Pressure", axis=1, inplace=True)

        df.dropna(inplace=True)

        st.session_state.df = df
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(df.head())

# -------- TRAIN MODELS --------
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare 5 Models")
    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()

        encoders = {}
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        X = df.drop("Sleep Disorder", axis=1)
        y = df["Sleep Disorder"]

        smt = SMOTETomek(random_state=SEED)
        X, y = smt.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=SEED
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        st.info("⏳ Training 5 models...")

        models = {
            "SVM": SVC(C=1, kernel="rbf", probability=True, random_state=SEED),
            "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=20, random_state=SEED),
            "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED),
            "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=SEED),
            "ANN": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED)
        }

        scores = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test)) * 100
            scores.append([name, round(acc, 4)])

        score_df = pd.DataFrame(scores, columns=["Model", "Accuracy"])
        st.table(score_df)

        best = score_df.loc[score_df["Accuracy"].idxmax()]
        st.success(f"🏆 Best Model: {best['Model']} ({best['Accuracy']}%)")

        st.session_state.best_model = models[best["Model"]]
        st.session_state.scaler = scaler
        st.session_state.encoders = encoders
        st.session_state.order = list(X.columns)

        cm = confusion_matrix(y_test, st.session_state.best_model.predict(X_test))
        fig, ax = plt.subplots()
        ax.imshow(cm)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        plt.colorbar()
        st.pyplot(fig)

        if st.button("💾 Save Best Model"):
            save_model(st.session_state.best_model, scaler, encoders, list(X.columns))
            st.success("✅ Best model saved!")

# -------- PREDICT DISORDER --------
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")
    if "best_model" not in st.session_state:
        st.warning("Train model first!")
    else:
        mode = st.radio("Prediction Mode", ["Manual Input", "Bulk Prediction"])

        if mode == "Manual Input":
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 1, 100, 25)
            occ = st.selectbox("Occupation", ["Engineer", "Doctor", "Teacher", "Manager", "Nurse", "Student"])
            sleep = st.slider("Sleep Duration", 3.0, 12.0, 7.0)
            q = st.slider("Quality of Sleep", 1, 10, 7)
            act = st.slider("Activity Level", 0, 100, 50)
            stress = st.slider("Stress Level", 1, 10, 5)
            bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese", "Underweight"])
            sys = st.slider("Systolic BP", 80, 180, 120)
            dia = st.slider("Diastolic BP", 50, 120, 80)
            heart = st.slider("Heart Rate", 40, 120, 70)
            steps = st.slider("Daily Steps", 0, 20000, 5000)

            data = pd.DataFrame([{
                "Gender": gender, "Age": age, "Occupation": occ,
                "Sleep Duration": sleep, "Quality of Sleep": q,
                "Physical Activity Level": act, "Stress Level": stress,
                "BMI Category": bmi, "Systolic_BP": sys,
                "Diastolic_BP": dia, "Heart Rate": heart,
                "Daily Steps": steps
            }])

            for col, le in st.session_state.encoders.items():
                if col in data.columns:
                    data[col] = data[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    data[col] = le.transform(data[col])

            data = data[st.session_state.order]
            scaled = st.session_state.scaler.transform(data)

            if st.button("🔮 Predict"):
                result = st.session_state.best_model.predict(scaled)[0]
                out = st.session_state.encoders["Sleep Disorder"].inverse_transform([result])[0]
                st.success(f"🩺 Predicted Disorder: {out}")

        else:
            file = st.file_uploader("Upload CSV without 'Sleep Disorder'", type=["csv"])
            if file:
                df = pd.read_csv(file)

                if "Blood Pressure" in df.columns:
                    df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
                    df.drop("Blood Pressure", axis=1, inplace=True)

                df.dropna(inplace=True)

                for col, le in st.session_state.encoders.items():
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        df[col] = le.transform(df[col])

                df = df[st.session_state.order]
                scaled = st.session_state.scaler.transform(df)
                pred = st.session_state.best_model.predict(scaled)
                df["Prediction"] = st.session_state.encoders["Sleep Disorder"].inverse_transform(pred)

                st.success("✅ Predictions completed!")
                st.dataframe(df.head())

                csv_file = df.to_csv(index=False)
                st.download_button("⬇ Download Predictions", csv_file, "sleep_predictions.csv")

# -------- INTERPRETABILITY --------
elif page == "📊 Interpretability":
    st.title("📊 Feature Importance")
    if "best_model" not in st.session_state:
        st.warning("Train model first!")
    else:
        df = st.session_state.df.copy()
        X = df[st.session_state.order]
        y = df["Sleep Disorder"]

        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=st.session_state.order)

        result = permutation_importance(
            st.session_state.best_model, X, y, n_repeats=10, random_state=SEED, scoring="accuracy"
        )

        idx = result.importances_mean.argsort()[::-1]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(np.array(st.session_state.order)[idx], result.importances_mean[idx])
        ax.set_xlabel("Importance Score")
        ax.set_title("Permutation Feature Importance")
        st.pyplot(fig)

        st.success("✅ Done!")
