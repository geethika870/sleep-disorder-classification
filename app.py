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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
st.sidebar.title("⚙ Navigation")
page = st.sidebar.radio("Go to:", ["📂 Upload Dataset", "🚀 Train Models", "🔮 Predict Disorder", "📊 Interpretability"])

# -------------------
# FIXED LGBMWrapper CLASS (correct dunder names)
# -------------------
class LGBMWrapper:
    def _init_(self, model):
        self.model = model

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def _getattr_(self, attr):
        return getattr(self.model, attr)

def save_model(best_model, scaler, label_encoders, feature_order):
    try:
        with open("best_model.pkl", "wb") as f:
            pickle.dump((best_model, scaler, label_encoders, feature_order), f)
        return True
    except Exception as e:
        st.error(f"Failed to save model: {e}")
        return False

def load_model_file():
    if os.path.exists("best_model.pkl"):
        try:
            with open("best_model.pkl", "rb") as f:
                return pickle.load(f)
        except Exception:
            return None, None, None, None
    return None, None, None, None

# 📂 Upload Dataset
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        # small cleaning convenience
        if "Person ID" in df.columns:
            df.drop("Person ID", axis=1, inplace=True)
        if "Blood Pressure" in df.columns:
            try:
                df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
                df.drop("Blood Pressure", axis=1, inplace=True)
            except Exception:
                st.warning("Could not parse 'Blood Pressure' column into two ints; leaving it as-is.")

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

        # ensure target exists
        if "Sleep Disorder" not in df.columns:
            st.error("Dataset must contain a 'Sleep Disorder' column as the target.")
        else:
            label_encoders = {}
            # encode object columns (including target)
            for col in df.select_dtypes(include="object").columns:
                le = LabelEncoder()
                try:
                    df[col] = le.fit_transform(df[col].astype(str))
                except Exception:
                    # fallback: replace NaNs then encode
                    df[col] = df[col].fillna("NA").astype(str)
                    df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

            X = df.drop("Sleep Disorder", axis=1)
            y = df["Sleep Disorder"]

            # handle class imbalance with SMOTETomek
            try:
                smt = SMOTETomek(random_state=SEED)
                X_res, y_res = smt.fit_resample(X, y)
            except Exception:
                st.warning("SMOTETomek failed or not appropriate — using original data.")
                X_res, y_res = X, y

            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2, stratify=y_res if len(np.unique(y_res)) > 1 else None, random_state=SEED
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            st.info("⏳ Training models...")

            models = {
                "SVM": SVC(C=1, kernel="rbf", probability=True, random_state=SEED),
                "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=20, random_state=SEED),
                "LightGBM": LGBMWrapper(LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED)),
                "CatBoost": CatBoostClassifier(iterations=300, verbose=0, random_state=SEED),
                "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=SEED),
                "ANN": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED),
            }

            results = {}
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    preds = model.predict(X_test_scaled)
                    results[name] = accuracy_score(y_test, preds)
                except Exception as e:
                    st.warning(f"Model {name} failed to train/predict: {e}")
                    results[name] = 0.0

            acc_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
            acc_df["Accuracy"] = (acc_df["Accuracy"] * 100).round(2)
            st.table(acc_df)

            best_idx = acc_df["Accuracy"].idxmax()
            best_model_name = acc_df.iloc[best_idx]["Model"]
            st.success(f"🏆 Best Model: {best_model_name}")

            st.session_state.best_model = models[best_model_name]
            st.session_state.scaler = scaler
            st.session_state.label_encoders = label_encoders
            st.session_state.feature_order = list(X.columns)

            try:
                cm = confusion_matrix(y_test, models[best_model_name].predict(X_test_scaled))
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not display confusion matrix: {e}")

            if st.button("💾 Save Best Model"):
                ok = save_model(models[best_model_name], scaler, label_encoders, list(X.columns))
                if ok:
                    st.success("✅ Model saved!")

# 🔮 Predict Disorder
elif page == "🔮 Predict Disorder":
    st.title("🔮 Predict Sleep Disorder")
    # try to load model from file if not in session
    if "best_model" not in st.session_state:
        loaded = load_model_file()
        if loaded[0] is not None:
            st.session_state.best_model, st.session_state.scaler, st.session_state.label_encoders, st.session_state.feature_order = loaded
            st.success("Loaded saved model from disk.")

    if "best_model" not in st.session_state:
        st.warning("Train or load a model first!")
    else:
        mode = st.radio("Prediction Mode", ["Manual Input", "Bulk Prediction"])

        if mode == "Manual Input":
            # keep form inputs consistent with feature_order if available, else fallback to a set
            if "feature_order" in st.session_state and st.session_state.feature_order:
                # attempt to create inputs based on typical names, else generic numeric inputs
                inputs = {}
                for feat in st.session_state.feature_order:
                    # guess type by name
                    if "gender" in feat.lower():
                        inputs[feat] = st.selectbox(feat, ["Male", "Female"], key=feat)
                    elif "bmi" in feat.lower() or "category" in feat.lower():
                        inputs[feat] = st.selectbox(feat, ["Normal", "Overweight", "Obese", "Underweight"], key=feat)
                    elif "age" in feat.lower():
                        inputs[feat] = st.slider(feat, 1, 100, 25, key=feat)
                    else:
                        # generic numeric input
                        inputs[feat] = st.number_input(feat, value=0.0, key=feat)
                if st.button("🔮 Predict"):
                    user_data = pd.DataFrame([inputs])
                    # ensure columns match exact order & types
                    for col, le in st.session_state.label_encoders.items():
                        if col in user_data.columns:
                            val = user_data[col].iloc[0]
                            val_str = str(val)
                            if val_str not in le.classes_:
                                # append unseen class to encoder
                                le.classes_ = np.append(le.classes_, val_str)
                            user_data[col] = le.transform(user_data[col].astype(str))
                    # re-order columns
                    user_data = user_data[st.session_state.feature_order]
                    scaled = st.session_state.scaler.transform(user_data.astype(float))
                    pred_num = st.session_state.best_model.predict(scaled)[0]
                    # decode target label if encoder exists
                    target_encoder = st.session_state.label_encoders.get("Sleep Disorder")
                    if target_encoder is not None:
                        try:
                            pred_label = target_encoder.inverse_transform([int(pred_num)])[0]
                        except Exception:
                            pred_label = pred_num
                    else:
                        pred_label = pred_num
                    st.success(f"🩺 Predicted Sleep Disorder: {pred_label}")

            else:
                st.info("No feature order known. Please upload/train first to use manual form.")

        else:
            file = st.file_uploader("Upload CSV without Sleep Disorder", type=["csv"])
            if file:
                new_df = pd.read_csv(file)
                if "Blood Pressure" in new_df.columns:
                    try:
                        new_df[["Systolic_BP", "Diastolic_BP"]] = new_df["Blood Pressure"].str.split("/", expand=True).astype(int)
                        new_df.drop("Blood Pressure", axis=1, inplace=True)
                    except Exception:
                        st.warning("Could not parse 'Blood Pressure' in uploaded CSV.")

                # transform categorical cols using saved encoders
                for col, le in st.session_state.label_encoders.items():
                    if col in new_df.columns:
                        # replace unseen values with first known class before transform
                        new_df[col] = new_df[col].apply(lambda x: x if str(x) in le.classes_ else le.classes_[0])
                        new_df[col] = le.transform(new_df[col].astype(str))

                # align columns to training feature order
                try:
                    new_df = new_df[st.session_state.feature_order]
                except Exception:
                    # if column names differ, attempt to use first N columns
                    new_df = new_df.iloc[:, :len(st.session_state.feature_order)]
                    new_df.columns = st.session_state.feature_order

                scaled = st.session_state.scaler.transform(new_df.astype(float))
                preds = st.session_state.best_model.predict(scaled)

                target_encoder = st.session_state.label_encoders.get("Sleep Disorder")
                if target_encoder is not None:
                    try:
                        preds_labels = target_encoder.inverse_transform(preds.astype(int))
                    except Exception:
                        preds_labels = preds
                else:
                    preds_labels = preds

                new_df["Predicted_Sleep_Disorder"] = preds_labels
                st.dataframe(new_df.head())

# 📊 Interpretability
elif page == "📊 Interpretability":
    st.title("📊 Model Interpretability - Feature Importance")
    if "best_model" not in st.session_state:
        st.warning("Train or load a model first!")
    elif "df" not in st.session_state:
        st.warning("Upload dataset first!")
    else:
        best_model = st.session_state.best_model
        scaler = st.session_state.scaler
        feature_order = st.session_state.feature_order
        df = st.session_state.df.copy()

        label_encoders = st.session_state.label_encoders
        # ensure categorical columns are encoded consistently for permutation importance
        for col in list(label_encoders.keys()):
            if col in df.columns and df[col].dtype == "object":
                le = label_encoders[col]
                df[col] = df[col].apply(lambda x: x if str(x) in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))

        X = df[feature_order]
        y = df["Sleep Disorder"]
        if y.dtype == "object" or not np.issubdtype(y.dtype, np.number):
            le_target = label_encoders.get("Sleep Disorder")
            if le_target:
                y_encoded = le_target.transform(y.astype(str))
            else:
                y_encoded = y
        else:
            y_encoded = y

        X_scaled = scaler.transform(X.astype(float))

        st.info("⏳ Calculating permutation importance...")

        try:
            result = permutation_importance(
                best_model, X_scaled, y_encoded,
                n_repeats=10, random_state=SEED, scoring="accuracy"
            )

            sorted_idx = result.importances_mean.argsort()[::-1]

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=result.importances_mean[sorted_idx], y=np.array(feature_order)[sorted_idx], ax=ax)
            ax.set_title("Permutation Feature Importance")
            ax.set_xlabel("Mean Importance")
            st.pyplot(fig)
            st.success("✅ Feature importance calculated successfully!")
        except Exception as e:
            st.error(f"Permutation importance failed: {e}")
