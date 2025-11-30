import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from streamlit_option_menu import option_menu
import base64

# -------- CSV DOWNLOAD BUTTON -------------
def download_csv(df, filename="dataset.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Sample CSV</a>'
    return href


# --------- TRAINING FUNCTION ---------------
def train_model(df):
    df = df.copy()

    # Label Encoding for categorical columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    X = df.drop("Sleep Disorder", axis=1)
    y = df["Sleep Disorder"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, acc


# -------- UI --------------
st.set_page_config(page_title="Sleep Disorder Predictor", layout="wide")

with st.sidebar:
    selected = option_menu(
        "Sleep Disorder App",
        ["Predict Manual", "Bulk Upload"],
        icons=["person", "cloud-upload"],
        menu_icon="heart",
        default_index=0
    )

st.title("😴 Sleep Disorder Prediction using Genetic Algorithm + ANN")

# -------------------- SAMPLE DATASET --------------------
sample_df = pd.DataFrame({
    "Age": [25, 40],
    "Gender": ["Male", "Female"],
    "Sleep Duration": [7, 5],
    "Stress Level": [3, 8],
    "Physical Activity Level": [50, 20],
    "Quality of Sleep": [7, 4],
    "Heart Rate": [72, 88],
    "Daily Steps": [8000, 3000],
    "Sleep Disorder": ["None", "Insomnia"]
})

st.markdown(download_csv(sample_df, "sample_sleep_disorder.csv"), unsafe_allow_html=True)

# ------------------- MANUAL PREDICTION ------------------
if selected == "Predict Manual":
    st.subheader("🧍 Manual Prediction")

    age = st.number_input("Age", 1, 100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    sleep_dur = st.slider("Sleep Duration (hrs)", 1, 12)
    stress = st.slider("Stress Level", 1, 10)
    activity = st.slider("Physical Activity Level", 0, 100)
    quality = st.slider("Quality of Sleep", 1, 10)
    heart = st.number_input("Heart Rate", 40, 150)
    steps = st.number_input("Daily Steps", 0, 20000)

    if st.button("Predict"):
        df = sample_df.copy()
        model, acc = train_model(df)

        user_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Sleep Duration": sleep_dur,
            "Stress Level": stress,
            "Physical Activity Level": activity,
            "Quality of Sleep": quality,
            "Heart Rate": heart,
            "Daily Steps": steps
        }])

        # encode
        le = LabelEncoder()
        for col in user_data.columns:
            if user_data[col].dtype == 'object':
                user_data[col] = le.fit_transform(user_data[col])

        prediction = model.predict(user_data)[0]

        st.success(f"🛌 Predicted Disorder: **{prediction}**")
        st.info(f"🔍 Model Accuracy: {acc*100:.2f}%")

# ---------------- BULK UPLOAD ------------------------
elif selected == "Bulk Upload":
    st.subheader("📑 Bulk Prediction")

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("📌 Uploaded Data:", df.head())

        if "Sleep Disorder" not in df.columns:
            st.error("❌ CSV must contain 'Sleep Disorder' column.")
        else:
            model, acc = train_model(df)
            preds = model.predict(df.drop("Sleep Disorder", axis=1))
            df["Predicted Disorder"] = preds

            st.success("🎉 Predictions Completed!")
            st.write(df)

            st.markdown(download_csv(df, "predictions.csv"), unsafe_allow_html=True)
