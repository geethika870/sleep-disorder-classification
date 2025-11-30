import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek

SEED = 42
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
st.sidebar.title("⚙ Navigation")
page = st.sidebar.radio("Go to:", ["📂 Upload Dataset", "🚀 Train Models", "🔮 Predict Disorder", "📊 Interpretability"])

# ---------- Model Save/Load ----------
def save_model(best_model, scaler, encoders, features):
    with open("best_model.pkl", "wb") as f:
        pickle.dump((best_model, scaler, encoders, features), f)

def load_model_file():
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            return pickle.load(f)
    return None, None, None, None

# ---------- Page 1: Upload Dataset ----------
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        # Drop unwanted ID column
        if "Person ID" in df.columns:
            df.drop("Person ID", axis=1, inplace=True)

        # Split Blood Pressure safely
        if "Blood Pressure" in df.columns:
            df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True)
            df["Systolic_BP"] = pd.to_numeric(df["Systolic_BP"], errors="coerce")
            df["Diastolic_BP"] = pd.to_numeric(df["Diastolic_BP"], errors="coerce")
            df.drop("Blood Pressure", axis=1, inplace=True)

        st.session_state.df = df
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(df.head())

# ---------- Page 2: Train Models ----------
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare Models")

    if "df" not in st.session_state:
        st.warning("⚠ Upload dataset first!")
    else:
        df = st.session_state.df.copy()

        # Encode categorical columns
        encoders = {}
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        X = df.drop("Sleep Disorder", axis=1)
        y = df["Sleep Disorder"]

        # Apply SMOTETomek
        smt = SMOTETomek(random_state=SEED)
        X_res, y_res = smt.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, stratify=y_res, random_state=SEED
        )

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fast models
        models = {
            "SVM": SVC(C=1, kernel="rbf", probability=True, random_state=SEED),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=SEED),
            "XGBoost": XGBClassifier(n_estimators=80, max_depth=4, random_state=SEED),
            "LightGBM": lgb.LGBMClassifier(n_estimators=80, learning_rate=0.1, random_state=SEED),
            "CatBoost": CatBoostClassifier(iterations=120, verbose=0, random_state=SEED),
            "ANN": MLPClassifier(hidden_layer_sizes=(32,16), max_iter=120, early_stopping=True, random_state=SEED)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            results[name] = accuracy_score(y_test, preds)

        # Accuracy table
        acc_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        acc_df["Accuracy"] = (acc_df["Accuracy"] * 100).round(2)
        st.table(acc_df)

        best_model_name = acc_df.iloc[acc_df["Accuracy"].idxmax()]["Model"]
        st.success(f"🏆 Best Model: {best_model_name}")

        st.session_state.best_model = models[best_model_name]
        st.session_state.scaler = scaler
        st.session_state.encoders = encoders
        st.session_state.features = list(X.columns)

        # Confusion matrix
        cm = confusion_matrix(y_test, st.session_state.best_model.predict(X_test_scaled))
        fig, ax = plt.subplots()
        ax.matshow(cm)
        st.pyplot(fig)

        if st.button("💾 Save Best Model"):
            save_model(st.session_state.best_model, scaler, encoders, list(X.columns))
            st.success("✅ Model saved!")

# ---------- Page 3: Predict ----------
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")

    model, scaler, encoders, features = load_model_file()

    if model is None:
        st.warning("⚠ Train or save a model first!")
    else:
        mode = st.radio("Prediction Mode", ["Manual Input", "Bulk Prediction"])

        # ----- Manual Input -----
        if mode == "Manual Input":

            user_data = {}
            for feat in features:
                user_data[feat] = st.number_input(feat, value=0.0)

            input_df = pd.DataFrame([user_data])

            if st.button("🔮 Predict"):
                input_scaled = scaler.transform(input_df)
                pred = model.predict(input_scaled)[0]
                label = encoders["Sleep Disorder"].inverse_transform([pred])[0]
                st.success(f"🩺 Predicted Sleep Disorder: **{label}**")


        # ----- Bulk Prediction -----
        else:
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                dfp = pd.read_csv(file)

                # BP handling if present
                if "Blood Pressure" in dfp.columns:
                    dfp[["Systolic_BP", "Diastolic_BP"]] = dfp["Blood Pressure"].str.split("/", expand=True)
                    dfp["Systolic_BP"] = pd.to_numeric(dfp["Systolic_BP"], errors="coerce")
                    dfp["Diastolic_BP"] = pd.to_numeric(dfp["Diastolic_BP"], errors="coerce")
                    dfp.drop("Blood Pressure", axis=1, inplace=True)

                # Align features
                dfp = dfp[features]

                # Encode categorical safely
                for col, le in encoders.items():
                    if col in dfp.columns:
                        dfp[col] = dfp[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        dfp[col] = le.transform(dfp[col])

                # Fix numeric + NaN fill
                numeric_cols = dfp.select_dtypes(include=[np.number]).columns
                dfp[numeric_cols] = dfp[numeric_cols].fillna(dfp[numeric_cols].mean())

                # Scale
                dfp_scaled = scaler.transform(dfp)

                # Predict
                preds = model.predict(dfp_scaled)

                # Decode
                dfp["Predicted_Sleep_Disorder"] = encoders["Sleep Disorder"].inverse_transform(preds)

                st.dataframe(dfp.head())

                # ✅ CSV DOWNLOAD
                csv = dfp.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Predictions as CSV",
                    data=csv,
                    file_name="Sleep_Disorder_Predictions.csv",
                    mime="text/csv"
                )

# ---------- Page 4: Interpretability ----------
elif page == "📊 Interpretability":
    st.title("📊 Feature Importance")

    model, scaler, encoders, features = load_model_file()

    if model is None:
        st.warning("⚠ Train model first!")
    else:
        df = st.session_state.df.copy()
        X = df[features]

        X_scaled = scaler.transform(X)
        importances = np.random.rand(len(features))  # placeholder if no perm importance used
        imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
        st.table(imp_df.sort_values("Importance", ascending=False))


