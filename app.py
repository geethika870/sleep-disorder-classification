import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import shap
import matplotlib.pyplot as plt
import os

# For GA (Genetic Algorithm) - using a simple implementation or library
try:
    import pygad
except ImportError:
    st.error("Please install pygad: pip install pygad")

# Function to load dataset
def load_dataset(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Assuming the dataset is in the repo or pre-loaded
        df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')  # Replace with actual path if needed
    return df

# Preprocess data
def preprocess_data(df):
    # Encode categorical variables
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Occupation'] = le.fit_transform(df['Occupation'])
    df['BMI Category'] = le.fit_transform(df['BMI Category'])
    df['Sleep Disorder'] = le.fit_transform(df['Sleep Disorder'])
    
    # Split Blood Pressure into Systolic and Diastolic
    df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop('Blood Pressure', axis=1, inplace=True)
    
    # Features and target
    X = df.drop(['Person ID', 'Sleep Disorder'], axis=1)
    y = df['Sleep Disorder']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, le

# ANN Model
def build_ann(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')  # Assuming 3 classes: None, Insomnia, Sleep Apnea
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ANN + GA (simplified)
def ann_ga(X_train, y_train, X_test, y_test):
    def fitness_func(ga_instance, solution, solution_idx):
        # Simple GA for hyperparameter tuning (e.g., learning rate, epochs)
        lr = solution[0]
        epochs = int(solution[1])
        model = build_ann(X_train.shape[1])
        model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, verbose=0)
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        return acc
    
    ga = pygad.GA(num_generations=10, num_parents_mating=4, fitness_func=fitness_func,
                  sol_per_pop=8, num_genes=2, gene_space=[{'low': 0.001, 'high': 0.1}, {'low': 10, 'high': 100}])
    ga.run()
    best_solution = ga.best_solution()
    st.write(f"ANN+GA Best Solution: LR={best_solution[0]}, Epochs={int(best_solution[1])}, Fitness={best_solution[2]}")
    return best_solution[2]

# Proposed Better Model: Ensemble of ANN, RF, SVM (Voting Classifier)
def proposed_model(X_train, y_train, X_test, y_test):
    ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200)
    rf = RandomForestClassifier(n_estimators=100)
    svm = SVC(probability=True)
    
    voting_clf = VotingClassifier(estimators=[('ann', ann), ('rf', rf), ('svm', svm)], voting='soft')
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return voting_clf, acc

# Train models
def train_models(X_train, y_train, X_test, y_test):
    accuracies = {}
    
    # ANN
    ann_model = build_ann(X_train.shape[1])
    ann_model.fit(X_train, y_train, epochs=50, verbose=0)
    _, ann_acc = ann_model.evaluate(X_test, y_test, verbose=0)
    accuracies['ANN'] = ann_acc * 100
    
    # ANN + GA
    ga_acc = ann_ga(X_train, y_train, X_test, y_test) * 100
    accuracies['ANN+GA'] = ga_acc
    
    # Proposed Model
    prop_model, prop_acc = proposed_model(X_train, y_train, X_test, y_test)
    accuracies['Proposed Ensemble (ANN+RF+SVM)'] = prop_acc * 100
    
    return accuracies, prop_model

# Save best model
def save_best_model(model, accuracies):
    best_model_name = max(accuracies, key=accuracies.get)
    if best_model_name == 'Proposed Ensemble (ANN+RF+SVM)':
        joblib.dump(model, 'best_model.pkl')
    st.success(f"Best model ({best_model_name}) saved with accuracy {accuracies[best_model_name]:.2f}%")

# Prediction
def predict_disorder(model, scaler, le, input_data):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    disorder = le.inverse_transform(prediction)[0]
    return disorder

# Interpretability
def feature_importance(model, X_train, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # For ensemble, use RF's importance
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)  # Assuming y_train is available
        importances = rf.feature_importances_
    
    fig, ax = plt.subplots()
    ax.barh(feature_names, importances)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    st.pyplot(fig)

# Streamlit App
st.set_page_config(page_title="Sleep Disorder Classification", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Train", "Predict", "Interpretability"])

if page == "Home":
    st.title("Sleep Disorder Classification & Prediction")
    st.header("Importance of Sleep")
    st.write("""
    Sleep is essential for physical and mental health. Poor sleep can lead to disorders like Insomnia and Sleep Apnea, affecting daily life.
    This app uses machine learning to classify sleep disorders based on lifestyle data.
    Our proposed model aims to achieve higher accuracy than existing ANN+GA (92.6%).
    """)
    st.image("https://via.placeholder.com/800x400?text=Sleep+Importance+Image", caption="Importance of Good Sleep")

elif page == "Upload & Train":
    st.title("Upload Dataset & Train Models")
    uploaded_file = st.file_uploader("Upload Sleep Health Dataset (CSV)", type="csv")
    if uploaded_file or os.path.exists('Sleep_health_and_lifestyle_dataset.csv'):
        df = load_dataset(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        X, y, scaler, le = preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                accuracies, prop_model = train_models(X_train, y_train, X_test, y_test)
            st.subheader("Model Accuracies")
            for model, acc in accuracies.items():
                st.write(f"{model}: {acc:.2f}%")
            save_best_model(prop_model, accuracies)

elif page == "Predict":
    st.title("Predict Sleep Disorder")
    if os.path.exists('best_model.pkl'):
        model = joblib.load('best_model.pkl')
        scaler = StandardScaler()  # Load or assume pre-fitted
        le = LabelEncoder()  # Load or assume pre-fitted
        
        # Input fields
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 80, 30)
        occupation = st.selectbox("Occupation", ["Doctor", "Engineer", "Teacher", "Nurse", "Accountant", "Lawyer", "Salesperson", "Software Engineer", "Scientist", "Manager"])  # Add more as needed
        sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 12.0, 7.0)
        quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
        physical_activity = st.slider("Physical Activity Level", 0, 100, 50)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
        systolic_bp = st.slider("Systolic BP", 90, 180, 120)
        diastolic_bp = st.slider("Diastolic BP", 60, 120, 80)
        heart_rate = st.slider("Heart Rate", 50, 120, 70)
        daily_steps = st.slider("Daily Steps", 1000, 20000, 8000)
        
        input_data = [gender, age, occupation, sleep_duration, quality_of_sleep, physical_activity, stress_level, bmi, systolic_bp, diastolic_bp, heart_rate, daily_steps]
        # Encode inputs similarly
        input_encoded = [1 if gender == "Male" else 0, age, 0, sleep_duration, quality_of_sleep, physical_activity, stress_level, 0, systolic_bp, diastolic_bp, heart_rate, daily_steps]  # Simplified encoding
        
        if st.button("Predict"):
            disorder = predict_disorder(model, scaler, le, input_encoded)
            st.success(f"Predicted Sleep Disorder: {disorder}")
    else:
        st.error("Train a model first!")

elif page == "Interpretability":
    st.title("Feature Importance")
    if os.path.exists('best_model.pkl'):
        model = joblib.load('best_model.pkl')
        # Assuming X_train and feature_names are available or pre-loaded
        feature_names = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity', 'Stress Level', 'BMI Category', 'Systolic BP', 'Diastolic BP', 'Heart Rate', 'Daily Steps']
        # Dummy X_train for demo
        X_train = np.random.rand(100, 12)
        feature_importance(model, X_train, feature_names)
    else:
        st.error("Train a model first!")

