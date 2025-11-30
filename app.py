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
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
st.sidebar.title("⚙ Navigation")
page = st.sidebar.radio("Go to:", ["📂 Upload Dataset", "🚀 Train Models", "🔮 Predict Disorder", "📊 Interpretability"])

# ⚡ ULTRA-FAST LGBMWrapper 
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

def save_model(best_model, scaler, label_encoders, feature_order):
    with open("best_model.pkl", "wb") as f:
        pickle.dump((best_model, scaler, label_encoders, feature_order), f)

def load_model_file():
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            return pickle.load(f)
    return None, None, None, None

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

# 🚀 Train Models - ULTRA FAST VERSION ⚡
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare Models")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload dataset first!")
    else:
        # ⚡ Progress bar & fast training
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        df = st.session_state.df.copy()
        status_text.text("🔄 Encoding categorical variables...")
        progress_bar.progress(10)

        label_encoders = {}
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        progress_bar.progress(20)

        X = df.drop("Sleep Disorder", axis=1)
        y = df["Sleep Disorder"]
        
        # ⚡ SKIP SMOTE for speed (optional toggle)
        if st.checkbox("🚀 Skip SMOTE (3x faster)", value=True):
            X_res, y_res = X, y
        else:
            status_text.text("🔄 Balancing data with SMOTETomek...")
            smt = SMOTETomek(random_state=SEED)
            X_res, y_res = smt.fit_resample(X, y)
        progress_bar.progress(30)

        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, stratify=y_res, random_state=SEED
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        progress_bar.progress(40)

        # ⚡ ULTRA-FAST MODEL CONFIGS
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1),
            "LightGBM": LGBMWrapper(LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED, n_jobs=-1, verbosity=-1)),
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED, n_jobs=-1, eval_metric="mlogloss"),
            "CatBoost": CatBoostClassifier(iterations=100, verbose=0, random_state=SEED, thread_count=-1),
        }

        results = {}
        model_names = list(models.keys())
        
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f"🚀 Training {name}... ({i+1}/{len(models)})")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            results[name] = accuracy_score(y_test, y_pred)
            progress_bar.progress(40 + (60 * (i+1) / len(models)))

        acc_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        acc_df["Accuracy"] = (acc_df["Accuracy"] * 100).round(2)
        st.table(acc_df)

        best_model_name = acc_df.loc[acc_df["Accuracy"].idxmax(), "Model"]
        st.success(f"🏆 Best Model: {best_model_name}")

        st.session_state.best_model = models[best_model_name]
        st.session_state.scaler = scaler
        st.session_state.label_encoders = label_encoders
        st.session_state.feature_order = list(X.columns)

        # ⚡ Fast confusion matrix
        cm = confusion_matrix(y_test, models[best_model_name].predict(X_test_scaled))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        if st.button("💾 Save Best Model"):
            save_model(models[best_model_name], scaler, label_encoders, list(X.columns))
            st.success("✅ Model saved!")
        
        progress_bar.progress(100)
        status_text.text("✅ Training complete!")

# 🔮 Predict Disorder
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")
    if "best_model" not in st.session_state:
        st.warning("⚠️ Please train a model first!")
    else:
        mode = st.radio("Prediction Mode", ["Manual Input", "Bulk Prediction"])

        if mode == "Manual Input":
            col1, col2, col3 = st.columns(3)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.slider("Age", 1, 100, 25)
            with col2:
                occupation = st.selectbox("Occupation", ["Software Engineer", "Doctor", "Nurse", "Teacher", "Manager", "Student"])
                sleep_dur = st.slider("Sleep Duration (hrs)", 3.0, 12.0, 7.0)
            with col3:
                q_sleep = st.slider("Quality of Sleep", 1, 10, 7)
                phys_act = st.slider("Physical Activity (%)", 0, 100, 50)

            col4, col5, col6 = st.columns(3)
            with col4:
                stress = st.slider("Stress Level", 1, 10, 5)
                bmi_cat = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese", "Underweight"])
            with col5:
                sys_bp = st.slider("Systolic BP", 80, 180, 120)
            with col6:
                dia_bp = st.slider("Diastolic BP", 50, 120, 80)
                hr = st.slider("Heart Rate (bpm)", 40, 120, 70)
                steps = st.slider("Daily Steps", 0, 20000, 5000)

            user_data = pd.DataFrame([{
                "Gender": gender, "Age": age, "Occupation": occupation,
                "Sleep Duration": sleep_dur, "Quality of Sleep": q_sleep,
                "Physical Activity Level": phys_act, "Stress Level": stress,
                "BMI Category": bmi_cat, "Systolic_BP": sys_bp,
                "Diastolic_BP": dia_bp, "Heart Rate": hr, "Daily Steps": steps
            }])

            for col, le in st.session_state.label_encoders.items():
                if col in user_data.columns:
                    if user_data[col].iloc[0] not in le.classes_:
                        le.classes_ = np.append(le.classes_, user_data[col].iloc[0])
                    user_data[col] = le.transform(user_data[col])

            user_data = user_data[st.session_state.feature_order]

            if st.button("🔮 Predict Disorder", type="primary"):
                scaled = st.session_state.scaler.transform(user_data)
                pred_num = st.session_state.best_model.predict(scaled)[0]
                target_encoder = st.session_state.label_encoders["Sleep Disorder"]
                pred_label = target_encoder.inverse_transform([pred_num])[0]
                st.success(f"🩺 **Predicted Sleep Disorder: {pred_label}**")
                st.balloons()

        else:
            file = st.file_uploader("Upload CSV (without Sleep Disorder column)", type=["csv"])
            if file:
                new_df = pd.read_csv(file)
                if "Blood Pressure" in new_df.columns:
                    new_df[["Systolic_BP", "Diastolic_BP"]] = new_df["Blood Pressure"].str.split("/", expand=True).astype(int)
                    new_df.drop("Blood Pressure", axis=1, inplace=True)

                for col, le in st.session_state.label_encoders.items():
                    if col in new_df.columns:
                        new_df[col] = new_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        new_df[col] = le.transform(new_df[col])

                new_df = new_df[st.session_state.feature_order]
                scaled = st.session_state.scaler.transform(new_df)
                preds = st.session_state.best_model.predict(scaled)

                target_encoder = st.session_state.label_encoders["Sleep Disorder"]
                preds_labels = target_encoder.inverse_transform(preds)

                new_df["Predicted_Sleep_Disorder"] = preds_labels
                st.dataframe(new_df)
                st.download_button("💾 Download Predictions", new_df.to_csv(index=False), "predictions.csv")

# 📊 Interpretability
elif page == "📊 Interpretability":
    st.title("📊 Model Interpretability")
    if "best_model" not in st.session_state:
        st.warning("⚠️ Please train a model first!")
    elif "df" not in st.session_state:
        st.warning("⚠️ Please upload dataset first!")
    else:
        best_model = st.session_state.best_model
        scaler = st.session_state.scaler
        feature_order = st.session_state.feature_order
        df = st.session_state.df.copy()

        label_encoders = st.session_state.label_encoders
        for col in label_encoders:
            if col in df.columns and df[col].dtype == "object":
                le = label_encoders[col]
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])

        X = df[feature_order]
        y = df["Sleep Disorder"]
        if y.dtype == "object":
            le_target = label_encoders.get("Sleep Disorder")
            y_encoded = le_target.transform(y) if le_target else y
        else:
            y_encoded = y

        X_scaled = scaler.transform(X)

        with st.spinner("⏳ Calculating feature importance..."):
            result = permutation_importance(
                best_model, X_scaled, y_encoded,
                n_repeats=5, random_state=SEED, scoring="accuracy"  # Reduced repeats
            )

        sorted_idx = result.importances_mean.argsort()[::-1]
        importance_df = pd.DataFrame({
            'Feature': np.array(feature_order)[sorted_idx],
            'Importance': result.importances_mean[sorted_idx]
        })

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=importance_df, y='Feature', x='Importance', ax=ax)
        ax.set_title("🔍 Permutation Feature Importance")
        ax.set_xlabel("Mean Importance Score")
        st.pyplot(fig)

        st.subheader("📋 Top Features")
        st.dataframe(importance_df.head(10))
        st.success("✅ Analysis complete!")
