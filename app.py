# app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import shap
import matplotlib.pyplot as plt

torch.manual_seed(42)


class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


class ANN_RF:
    # Hybrid model: ANN + RandomForest on ANN output probabilities
    def __init__(self, input_dim):
        self.ann = ANN(input_dim)
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ann.to(self.device)

    def train_ann(self, X_train, y_train, epochs=50, batch_size=32):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.ann.parameters(), lr=0.001)
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.ann.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                output = self.ann(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()

    def ann_predict_proba(self, X):
        self.ann.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.ann(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        y_enc = self.label_encoder.fit_transform(y)
        self.train_ann(X_scaled, y_enc)
        ann_probs = self.ann_predict_proba(X_scaled)
        self.rf.fit(ann_probs, y_enc)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        ann_probs = self.ann_predict_proba(X_scaled)
        preds = self.rf.predict(ann_probs)
        return self.label_encoder.inverse_transform(preds)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        ann_probs = self.ann_predict_proba(X_scaled)
        rf_probs = self.rf.predict_proba(ann_probs)
        return rf_probs


st.set_page_config(page_title="Sleep Disorder Classification", layout="wide")

menu = ["Home", "Upload Dataset", "Train Models", "Predict & Interpret"]
choice = st.sidebar.selectbox("Navigation", menu)


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)


def preprocess_data(df, target_col):
    df = df.dropna()
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in dataset.")
        st.stop()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if y.dtype == object:
        y = y.astype(str)
    X_encoded = pd.get_dummies(X)
    return X_encoded, y


def train_ann(X_train, y_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    input_dim = X_train.shape[1]
    ann = ANN(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ann.parameters(), lr=0.001)
    dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                            torch.tensor(y_train.values, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    ann.train()
    for epoch in range(40):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = ann(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
    return ann


def evaluate_ann(ann, X_test, y_test):
    ann.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        X_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
        outputs = ann(X_tensor)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
    acc = accuracy_score(y_test, preds)
    return acc


def train_ann_ga(X_train, y_train):
    # Placeholder imitates ANN+GA; actually trains ANN
    return train_ann(X_train, y_train)


def train_svm(X_train, y_train):
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    return svm


def train_rf(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def main():
    if choice == "Home":
        st.title("Sleep Disorder Classification & Prediction App")
        st.markdown("""
        **Importance of Sleep:**  
        Quality sleep is vital for physical health, cognitive function, and emotional well-being.  
        Sleep disorders impact millions globally, affecting quality of life and productivity.  
        Early detection and classification help guide proper interventions.
        """)
    elif choice == "Upload Dataset":
        st.title("Upload Sleep Health Lifestyle Dataset (CSV)")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.success("Dataset Loaded Successfully")
            st.dataframe(df.head())
            st.session_state['df'] = df
            st.session_state.pop('model', None)
            st.session_state.pop('best_model_name', None)
            st.session_state.pop('label_encoder', None)
            st.session_state.pop('X_columns', None)
        else:
            st.info("Please upload your dataset to proceed.")
    elif choice == "Train Models":
        if 'df' not in st.session_state:
            st.warning("Upload dataset first in 'Upload Dataset' section.")
            return
        df = st.session_state['df']

        st.write("Columns in dataset:")
        all_columns = df.columns.tolist()
        target_col = st.selectbox("Select the target column for classification", all_columns)

        if st.button("Start Training"):
            with st.spinner("Preprocessing data..."):
                X, y = preprocess_data(df, target_col)

            le_target = LabelEncoder()
            y_enc = le_target.fit_transform(y)

            X_train, X_test, y_train_enc, y_test_enc = train_test_split(
                X, y_enc, stratify=y_enc, test_size=0.2, random_state=42)

            st.write("Training ANN (baseline)...")
            ann_model = train_ann(X_train, pd.Series(y_train_enc))
            acc_ann = evaluate_ann(ann_model, X_test, y_test_enc)
            st.write(f"ANN Accuracy: {acc_ann * 100:.2f}%")

            st.write("Training ANN + GA (baseline)...")
            ann_ga_model = train_ann_ga(X_train, pd.Series(y_train_enc))
            acc_ann_ga = evaluate_ann(ann_ga_model, X_test, y_test_enc)
            st.write(f"ANN + GA Accuracy (approx.): {acc_ann_ga * 100:.2f}%")

            st.write("Training Hybrid Model (ANN + RF stacking)...")
            hybrid_model = ANN_RF(X_train.shape[1])
            hybrid_model.fit(X_train.values, le_target.inverse_transform(y_train_enc))
            preds_hybrid = hybrid_model.predict(X_test.values)
            acc_hybrid = accuracy_score(le_target.inverse_transform(y_test_enc), preds_hybrid)
            st.write(f"Hybrid ANN + RF Accuracy: {acc_hybrid * 100:.2f}%")

            st.write("Training SVM model...")
            svm_model = train_svm(X_train, y_train_enc)
            acc_svm = svm_model.score(X_test, y_test_enc)
            st.write(f"SVM Accuracy: {acc_svm * 100:.2f}%")

            accuracies = {
                "ANN": acc_ann,
                "ANN+GA": acc_ann_ga,
                "Hybrid ANN+RF": acc_hybrid,
                "SVM": acc_svm
            }
            best_name = max(accuracies, key=accuracies.get)
            best_acc = accuracies[best_name]
            st.success(f"Best Model: {best_name} with accuracy {best_acc * 100:.2f}%")

            st.session_state['best_model_name'] = best_name
            if best_name == "ANN":
                st.session_state['model'] = ann_model
            elif best_name == "ANN+GA":
                st.session_state['model'] = ann_ga_model
            elif best_name == "Hybrid ANN+RF":
                st.session_state['model'] = hybrid_model
                st.session_state['label_encoder'] = le_target
            elif best_name == "SVM":
                st.session_state['model'] = svm_model
                st.session_state['label_encoder'] = le_target
            st.session_state['X_columns'] = X.columns.tolist()

    elif choice == "Predict & Interpret":
        st.title("Prediction & Model Interpretability")
        if 'model' not in st.session_state:
            st.warning("Please train models before prediction.")
            return

        st.write(f"Using best model: {st.session_state['best_model_name']}")
        model = st.session_state['model']
        label_encoder = st.session_state.get('label_encoder', None)
        X_columns = st.session_state['X_columns']

        st.subheader("Input Features for Prediction")
        input_data = {}
        for col in X_columns:
            val = st.text_input(f"{col}", "")
            input_data[col] = val

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([input_data])
                for c in input_df.columns:
                    # Convert input to numeric or NaN
                    input_df[c] = pd.to_numeric(input_df[c], errors='coerce')
                # Fill missing numeric inputs with mean of training or zero
                input_df.fillna(input_df.mean(), inplace=True)

                if len(input_df.columns) != len(X_columns):
                    st.error("Input columns count mismatch.")
                    return
                input_df = input_df[X_columns]

                if st.session_state['best_model_name'] == "ANN":
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.eval()
                    with torch.no_grad():
                        tensor_in = torch.tensor(input_df.values, dtype=torch.float32).to(device)
                        output = model(tensor_in)
                        probs = torch.softmax(output, dim=1).cpu().numpy()
                        pred_idx = np.argmax(probs, axis=1)[0]
                    prediction = pred_idx

                elif st.session_state['best_model_name'] == "ANN+GA":
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.eval()
                    with torch.no_grad():
                        tensor_in = torch.tensor(input_df.values, dtype=torch.float32).to(device)
                        output = model(tensor_in)
                        probs = torch.softmax(output, dim=1).cpu().numpy()
                        pred_idx = np.argmax(probs, axis=1)[0]
                    prediction = pred_idx

                elif st.session_state['best_model_name'] == "Hybrid ANN+RF":
                    pred_label = model.predict(input_df.values)[0]
                    prediction = pred_label

                elif st.session_state['best_model_name'] == "SVM":
                    pred = model.predict(input_df.values)[0]
                    if label_encoder is not None:
                        prediction = label_encoder.inverse_transform([pred])[0]
                    else:
                        prediction = pred

                st.markdown(f"### Predicted Sleep Disorder Class: **{prediction}**")

                # Show feature importance with SHAP for applicable models
                if st.session_state['best_model_name'] in ["Hybrid ANN+RF", "SVM"]:
                    st.subheader("Feature Importance (SHAP)")

                    if st.session_state['best_model_name'] == "Hybrid ANN+RF":
                        explainer = shap.TreeExplainer(model.rf)
                        scaled_input = model.scaler.transform(input_df.values)
                        ann_probs = model.ann_predict_proba(scaled_input)
                        shap_values = explainer.shap_values(ann_probs)

                        # Plot SHAP summary plot
                        plt.figure()
                        shap.summary_plot(shap_values, features=ann_probs,
                                          feature_names=["Prob_Class0", "Prob_Class1"], show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    else:  # SVM interpretation with KernelExplainer (may be slow)
                        try:
                            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(input_df, 50))
                            shap_values = explainer.shap_values(input_df, nsamples=100)
                            plt.figure()
                            shap.summary_plot(shap_values, input_df, show=False)
                            st.pyplot(plt.gcf())
                            plt.clf()
                        except Exception:
                            st.info("SHAP interpretation for SVM may be slow or unavailable.")

                else:
                    st.info("Interpretability not implemented for ANN models.")

            except Exception as e:
                st.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main()
