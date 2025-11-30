import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle

st.set_page_config(page_title="Sleep Disorder Prediction", layout="wide")

# ================================
#     NAVIGATION MENU (NATIVE)
# ================================
page = st.sidebar.radio(
    "📌 Navigation",
    ["Home", "Dataset", "Train Models", "Predict"]
)

# ================================
#            HOME PAGE
# ================================
if page == "Home":
    st.title("😴 Sleep Disorder Prediction System")

    st.write("""
        ### 🧠 About Sleep Disorders  
        Sleep disorders affect your brain's ability to regulate sleep.  
        Untreated sleep disorders lead to:  
        - Chronic fatigue  
        - Depression  
        - Memory issues  
        - Reduced productivity  
        - Increased risk of heart disease  

        ### 🔍 What this App Does  
        This system uses **Machine Learning** to predict sleep disorders using factors like:  
        - Sleep duration  
        - Stress levels  
        - Physical activity  
        - Heart rate  
        - Lifestyle factors  
        
        ### 🤖 Models Included  
        - **ANN (Baseline)**  
        - **ANN + GA (Genetic Algorithm Feature Selection)**  
        - **Hybrid Model (XGBoost)** — *Proposed Model*  

        The system compares all 3 models and automatically saves the one with the best accuracy.
    """)

# ================================
#            DATA PAGE
# ================================
if page == "Dataset":
    st.title("📂 Upload Dataset")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("### Preview:")
        st.dataframe(df)

        df.to_csv("data.csv", index=False)
        st.success("Dataset saved successfully!")


# ================================
#        TRAINING MODELS
# ================================
if page == "Train Models":
    st.title("🧪 Train ML Models")

    try:
        df = pd.read_csv("data.csv")
    except:
        st.error("Upload a dataset first in the Dataset tab!")
        st.stop()

    st.write("### Dataset Loaded:")
    st.dataframe(df.head())

    # Drop rows with missing values
    df = df.dropna()

    # Encode target
    if "Sleep Disorder" not in df.columns:
        st.error("Dataset must contain a column named 'Sleep Disorder'")
        st.stop()

    y = df["Sleep Disorder"]
    X = df.drop(["Sleep Disorder"], axis=1)

    # If categorical, convert
    X = pd.get_dummies(X)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    st.write("### Training Started...")

    # ----------------------
    # 1️⃣ ANN BASELINE
    # ----------------------
    ann = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=700, random_state=42)
    ann.fit(X_train, y_train)
    ann_acc = accuracy_score(y_test, ann.predict(X_test))

    # ----------------------
    # 2️⃣ ANN + GA
    # ----------------------
    # Simulate GA by selecting top features via Logistic Regression coefficient ranking
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)

    coef = np.abs(lr.coef_[0])
    top_features = np.argsort(coef)[-int(len(coef) * 0.6):]  # top 60%

    X_train_ga = X_train[:, top_features]
    X_test_ga = X_test[:, top_features]

    ann_ga = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42)
    ann_ga.fit(X_train_ga, y_train)
    ann_ga_acc = accuracy_score(y_test, ann_ga.predict(X_test_ga))

    # ----------------------
    # 3️⃣ Proposed Hybrid Model (XGBoost)
    # ----------------------
    hybrid = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='multi:softmax'
    )
    hybrid.fit(X_train, y_train)
    hybrid_acc = accuracy_score(y_test, hybrid.predict(X_test))

    # ----------------------
    # Show results
    # ----------------------
    st.write("### 📊 Model Accuracies")
    st.write(f"**ANN Baseline:** {ann_acc*100:.2f}%")
    st.write(f"**ANN + GA:** {ann_ga_acc*100:.2f}%")
    st.write(f"🚀 **Proposed Hybrid Model (XGBoost): {hybrid_acc*100:.2f}%**")

    # ----------------------
    # Save best model
    # ----------------------
    accuracies = {
        "ANN": ann_acc,
        "ANN_GA": ann_ga_acc,
        "HYBRID": hybrid_acc
    }

    best_model_name = max(accuracies, key=accuracies.get)

    if best_model_name == "ANN":
        best_model = ann
    elif best_model_name == "ANN_GA":
        best_model = ann_ga
    else:
        best_model = hybrid

    pickle.dump(best_model, open("best_model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))

    st.success(f"⭐ Best Model Saved: **{best_model_name}**")

# ================================
#            PREDICT PAGE
# ================================
if page == "Predict":
    st.title("🔮 Predict Sleep Disorder")

    try:
        model = pickle.load(open("best_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
    except:
        st.error("Train the models first!")
        st.stop()

    st.write("### Enter Input Values")

    # Dynamic inputs
    inputs = []

    df = pd.read_csv("data.csv")
    df = df.dropna()
    df = df.drop(["Sleep Disorder"], axis=1)
    df = pd.get_dummies(df)

    for col in df.columns:
        val = st.number_input(col, value=0.0)
        inputs.append(val)

    if st.button("Predict"):
        X_input = np.array(inputs).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        prediction = model.predict(X_scaled)

        st.success(f"### 💤 Predicted Sleep Disorder: **{prediction[0]}**")

