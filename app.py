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
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(page_title="😴 Sleep Disorder Prediction", layout="wide")
page = st.sidebar.radio("⚙ Navigation", ["📂 Upload Dataset", "🚀 Train Models", "🔮 Predict", "📊 Interpretability"])

# Save & load model utilities
def save_model(best_model, scaler, encoders, features):
    with open("best_model.pkl", "wb") as f:
        pickle.dump((best_model, scaler, encoders, features), f)

def load_model_file():
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", "rb") as f:
            return pickle.load(f)
    return None, None, None, None

# 📂 Upload Dataset Page
if page == "📂 Upload Dataset":
    st.title("📂 Upload Sleep Dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

        # Clean & parse BP if present
        if "Blood Pressure" in df.columns:
            try:
                df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
                df.drop("Blood Pressure", axis=1, inplace=True)
            except:
                st.warning("Blood Pressure parsing failed — keeping original column.")

        st.session_state.df = df
        st.success("✅ Dataset uploaded!")
        st.dataframe(df.head())

# 🚀 Train & Compare 5 Models (Auto-save best)
elif page == "🚀 Train Models":
    st.title("🚀 Train and Compare 5 Models")

    if "df" not in st.session_state:
        st.warning("Upload dataset first")
    else:
        df = st.session_state.df.copy()

        if "Sleep Disorder" not in df.columns:
            st.error("Target column 'Sleep Disorder' not found")
        else:
            # Encode categoricals safely
            encoders = {}
            for col in df.select_dtypes(include="object").columns:
                le = LabelEncoder()
                df[col] = df[col].fillna("NA").astype(str)
                df[col] = le.fit_transform(df[col])
                encoders[col] = le

            # One-hot for trees to learn better splits
            X = df.drop("Sleep Disorder", axis=1)
            y = df["Sleep Disorder"].fillna(y).values if df["Sleep Disorder"].isnull().any() else df["Sleep Disorder"].values
            X = pd.get_dummies(X, drop_first=True)

            # Remove NaN from target if any
            if isinstance(y, pd.Series) and y.isnull().any():
                y = y.fillna("NA")
                y = LabelEncoder().fit_transform(y.astype(str))

            # Balance classes
            smt = SMOTETomek(random_state=SEED)
            X_res, y_res = smt.fit_resample(X, y)

            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=SEED)

            # Scale only for models that need it
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 5 Models with tuned LGBM
            models = {
                "SVM": SVC(C=2.0, kernel="rbf", probability=True, random_state=SEED),
                "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=20, random_state=SEED),
                "LightGBM": LGBMClassifier(
                    n_estimators=600,
                    learning_rate=0.03,
                    max_depth=25,
                    num_leaves=50,
                    min_child_samples=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=SEED
                ),
                "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.04, max_depth=10, eval_metric="mlogloss", random_state=SEED),
                "ANN": MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=700, random_state=SEED)
            }

            # Train & evaluate
            results = {}
            for name, model in models.items():
                try:
                    if name == "LightGBM" or name == "Random Forest" or name == "XGBoost":
                        model.fit(X_train, y_train)  # trees don't need scaling
                    else:
                        model.fit(X_train_scaled, y_train)

                    preds = model.predict(X_test if name in ["LightGBM","Random Forest","XGBoost"] else X_test_scaled)
                    results[name] = accuracy_score(y_test, preds)
                except Exception as e:
                    st.error(f"{name} training failed: {e}")

            # Show Accuracies
            acc_df = pd.DataFrame(list(results.items()), columns=["Model","Accuracy"])
            acc_df["Accuracy"] = (acc_df["Accuracy"] * 100).round(2)
            st.table(acc_df)

            # Pick best model
            best = acc_df.iloc[acc_df["Accuracy"].idxmax()]
            st.success(f"🏆 Best Model: {best['Model']} ({best['Accuracy']}%)")

            # Save to session & disk auto
            best_model = models[best["Model"]]
            st.session_state.best_model = best_model
            st.session_state.scaler = scaler
            st.session_state.encoders = encoders
            st.session_state.features = X_train.columns.tolist()
            save_model(best_model, scaler, encoders, st.session_state.features)
            st.info("💾 Best model saved automatically to disk.")

            # Confusion Matrix FIX
            try:
                cm = confusion_matrix(y_test, best_model.predict(X_test))
                fig, ax = plt.subplots()
                im = ax.imshow(cm)
                plt.colorbar(im)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, cm[i, j], ha="center", va="center")
                st.pyplot(fig)
            except:
                pass

# 🔮 Prediction Page (manual + bulk + CSV export)
elif page == "🔮 Predict":
    st.title("🔮 Predict Sleep Disorder")

    if "best_model" not in st.session_state:
        best_model, scaler, encoders, features = load_model_file()
        if best_model:
            st.session_state.best_model = best_model
            st.session_state.scaler = scaler
            st.session_state.encoders = encoders
            st.session_state.features = features
            st.success("✅ Loaded saved model from disk")

    if "best_model" in st.session_state:
        mode = st.radio("Prediction Mode", ["Manual","Bulk CSV"])

        if mode == "Manual":
            user = {}
            for f in st.session_state.features:
                user[f] = st.number_input(f, value=0.0)
            if st.button("Predict"):
                data = pd.DataFrame([user])[st.session_state.features]
                scaled = st.session_state.scaler.transform(data)
                pred = st.session_state.best_model.predict(scaled)[0]
                st.success(f"🩺 Predicted Disorder: {pred}")

        else:
            file = st.file_uploader("Upload CSV without target", type=["csv"])
            if file:
                new_df = pd.read_csv(file)
                for col, le in st.session_state.encoders.items():
                    if col in new_df.columns:
                        new_df[col] = new_df[col].fillna("NA").astype(str)
                        new_df[col] = new_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        new_df[col] = le.transform(new_df[col].astype(str))
                new_df = pd.get_dummies(new_df).reindex(columns=st.session_state.features, fill_value=0)
                scaled = st.session_state.scaler.transform(new_df)
                preds = st.session_state.best_model.predict(scaled)
                new_df["Predicted Disorder"] = preds
                st.dataframe(new_df.head())
                out = new_df.to_csv(index=False)
                st.download_button("📥 Download Predictions CSV", out, "predictions.csv", "text/csv")

    else:
        st.warning("Train a model first")

# 📊 Interpretability Page (safe float transform)
elif page == "📊 Interpretability":
    st.title("📊 Feature Importance")

    if "best_model" not in st.session_state:
        st.warning("Train model first")
    else:
        try:
            df = st.session_state.df.copy()
            df = pd.get_dummies(df).reindex(columns=st.session_state.features, fill_value=0)
            X = df[st.session_state.features]
            y = st.session_state.df["Sleep Disorder"].values
            X = X.fillna(0)
            result = permutation_importance(st.session_state.best_model, st.session_state.scaler.transform(X), y, n_repeats=5, random_state=SEED)
            imp = result.importances_mean
            imp_df = pd.DataFrame({"Feature": st.session_state.features, "Importance": imp})
            imp_df = imp_df.sort_values("Importance", ascending=False)
            st.bar_chart(imp_df.set_index("Feature"))
        except Exception as e:
            st.error(f"Interpretability failed: {e}")
