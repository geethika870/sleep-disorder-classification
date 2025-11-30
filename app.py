import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, random
import matplotlib.pyplot as plt
import seaborn as sns
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

def save_model(model, scaler, encoders, order):
    with open("best_model.pkl","wb") as f:
        pickle.dump((model, scaler, encoders, order), f)

def load_model():
    if os.path.exists("best_model.pkl"):
        return pickle.load(open("best_model.pkl","rb"))
    return None, None, None, None

# 📂 Upload
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        if "Person ID" in df.columns:
            df.drop("Person ID",axis=1,inplace=True)

        if "Blood Pressure" in df.columns:
            try:
                bp = df["Blood Pressure"].astype(str).str.split("/",expand=True).astype(int)
                df["Systolic_BP"], df["Diastolic_BP"] = bp[0], bp[1]
                df.drop("Blood Pressure",axis=1,inplace=True)
            except:
                st.warning("Could not split Blood Pressure")

        st.session_state.df = df
        st.success("✅ Dataset Uploaded")
        st.dataframe(df.head())

# 🚀 Train
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare Models")
    if "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        df = st.session_state.df.copy()

        # encode categories
        encoders = {}
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            le.fit(df[col])
            df[col] = le.transform(df[col])
            encoders[col] = le

        if "Sleep Disorder" not in df.columns:
            st.error("Target column missing")
        else:
            X = df.drop("Sleep Disorder",axis=1)
            y = df["Sleep Disorder"]

            try:
                smt = SMOTETomek(random_state=SEED)
                X, y = smt.fit_resample(X, y)
            except:
                st.warning("SMOTETomek failed")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            models = {
                "SVM": SVC(probability=True, random_state=SEED),
                "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
                "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED),
                "CatBoost": CatBoostClassifier(iterations=300, verbose=0, random_state=SEED),
                "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=SEED),
                "ANN": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=400, random_state=SEED)
            }

            results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    results[name] = accuracy_score(y_test, model.predict(X_test)) * 100
                except Exception as e:
                    st.warning(f"{name} failed: {e}")
                    results[name] = 0.0

            acc_df = pd.DataFrame({
                "Model": list(results.keys()),
                "Accuracy (%)": np.round(list(results.values()),2)
            })

            st.table(acc_df)

            best = acc_df.loc[acc_df["Accuracy (%)"].idxmax()]
            st.success(f"🏆 Best Model: {best['Model']} → {best['Accuracy (%)']}%")

            st.session_state.best_model = models[best["Model"]]
            st.session_state.scaler = scaler
            st.session_state.encoders = encoders
            st.session_state.feature_order = list(X.columns)

            # show class map
            le_target = encoders["Sleep Disorder"]
            class_names = le_target.inverse_transform(list(range(len(le_target.classes_))))
            class_display = {i:name for i,name in enumerate(class_names)}
            st.info("🩺 Class Mapping:")
            st.table(pd.DataFrame({"Encoded Label":list(class_display.keys()), "Disorder Name": list(class_display.values())}))

            save_model(st.session_state.best_model, scaler, encoders, st.session_state.feature_order)
            st.success("✅ Model Saved to Disk")

            cm = confusion_matrix(y_test, st.session_state.best_model.predict(X_test))
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            st.pyplot(fig)

# 🔮 Prediction
elif page == "🔮 Predict Disorder":
    st.title("🔮 Sleep Disorder Prediction")

    if "best_model" not in st.session_state:
        model, scaler, encoders, feature_order = load_model()
        if model:
            st.session_state.best_model = model
            st.session_state.scaler = scaler
            st.session_state.encoders = encoders
            st.session_state.feature_order = feature_order
            st.session_state.feature_order = feature_order
            st.success("✅ Loaded saved model from disk")

    if "best_model" not in st.session_state:
        st.warning("Train a model first!")
    else:
        mode = st.radio("Prediction Mode", ["Manual", "Bulk"])

        if mode == "Manual":
            user_input = {}
            st.subheader("Enter Feature Values")
            for col in st.session_state.feature_order:
                user_input[col] = st.number_input(col, value=0.0)

            if st.button("🔮 Predict"):
                df_input = pd.DataFrame([user_input])

                for col, le in st.session_state.encoders.items():
                    if col in df_input.columns:
                        df_input[col] = df_input[col].astype(str)
                        df_input[col] = df_input[col].apply(lambda x: x if x in le.classes_ else "NA")
                        if "NA" not in le.classes_:
                            le.classes_ = np.append(le.classes_, "NA")
                        df_input[col] = le.transform(df_input[col])

                df_input = df_input[st.session_state.feature_order]
                scaled = st.session_state.scaler.transform(df_input.astype(float))
                pred = st.session_state.best_model.predict(scaled)[0]
                le_target = st.session_state.encoders["Sleep Disorder"]
                label = le_target.inverse_transform([int(pred)])[0]
                st.success(f"🩺 Predicted Sleep Disorder: {label}")

        else:
            file = st.file_uploader("Upload prediction CSV", type=["csv"])
            if file:
                df_bulk = pd.read_csv(file)

                if "Blood Pressure" in df_bulk.columns:
                    try:
                        bp = df_bulk["Blood Pressure"].astype(str).str.split("/",expand=True).astype(int)
                        df_bulk["Systolic_BP"], df_bulk["Diastolic_BP"] = bp[0], bp[1]
                        df_bulk.drop("Blood Pressure",axis=1,inplace=True)
                    except:
                        st.warning("Could not split Blood Pressure")

                for col, le in st.session_state.encoders.items():
                    if col in df_bulk.columns:
                        df_bulk[col] = df_bulk[col].astype(str)
                        df_bulk[col] = df_bulk[col].apply(lambda x: x if x in le.classes_ else "NA")
                        if "NA" not in le.classes_:
                            le.classes_ = np.append(le.classes_, "NA")
                        df_bulk[col] = le.transform(df_bulk[col])

                try:
                    df_bulk = df_bulk[st.session_state.feature_order]
                except:
                    df_bulk = df_bulk.iloc[:, :len(st.session_state.feature_order)]
                    df_bulk.columns = st.session_state.feature_order

                scaled = st.session_state.scaler.transform(df_bulk.astype(float))
                preds = st.session_state.best_model.predict(scaled).astype(int)
                le_target = st.session_state.encoders["Sleep Disorder"]
                labels = le_target.inverse_transform(preds)
                df_bulk["Predicted Sleep Disorder"] = labels
                st.subheader("✅ Prediction Results")
                st.dataframe(df_bulk.head())

                csv = df_bulk.to_csv(index=False).encode('utf-8')
                st.download_button("⬇ Download Predictions CSV", data=csv, file_name="sleep_predictions.csv")

# 📊 Interpretability
else:
    st.title("📊 Feature Importance Analysis")
    if "df" not in st.session_state or "best_model" not in st.session_state:
        st.warning("Upload & train first!")
    else:
        df = st.session_state.df.copy()

        for col, le in st.session_state.encoders.items():
            if col in df.columns and df[col].dtype == "object":
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else "NA")
                if "NA" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "NA")
                df[col] = le.transform(df[col])

        X = df[st.session_state.feature_order]
        y = df["Sleep Disorder"]
        X_scaled = st.session_state.scaler.transform(X.astype(float))

        imp = permutation_importance(st.session_state.best_model, X_scaled, y, n_repeats=8,random_state=SEED)
        idx = imp.importances_mean.argsort()[::-1]

        fig, ax = plt.subplots()
        ax.barh(np.array(st.session_state.feature_order)[idx], imp.importances_mean[idx])
        st.pyplot(fig)
        st.success("✅ Feature Importance Calculated!")
