import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_wine

# Load dataset
wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
df['class'] = wine_data.target


# Helper functions for analysis and visualization
def visualize_data():
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Class Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='class', data=df, palette='rocket', ax=ax)
    ax.set_title('Class Distribution')
    st.pyplot(fig)

    st.subheader("Feature Correlation Matrix")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, ax=ax)
    st.pyplot(fig)


# Function to preprocess data and train models
def train_model(algorithm):
    # Splitting the dataset
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if algorithm == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif algorithm == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == "LightGBM":
        model = LGBMClassifier(n_estimators=100, random_state=42)

    # Train and predict
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=wine_data.target_names)

    return accuracy, conf_matrix, class_report, model


def display_results(algorithm):
    st.subheader(f"Results for {algorithm} Algorithm")
    accuracy, conf_matrix, class_report, model = train_model(algorithm)

    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write("**Classification Report**:")
    st.text(class_report)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=wine_data.target_names,
                yticklabels=wine_data.target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

    if algorithm != "KNN":
        st.subheader("Feature Importance")
        feature_importances = pd.Series(
            model.feature_importances_, index=wine_data.feature_names
        ).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=feature_importances, y=feature_importances.index, palette='viridis', ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)


# Streamlit App
st.title("Wine Quality Analysis")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Overview", "Machine Learning Models"])
algorithm = None

if page == "Machine Learning Models":
    st.sidebar.subheader("Select Algorithm")
    algorithm = st.sidebar.radio("Algorithm", ["KNN", "Random Forest", "LightGBM"])

# Main Page Content
if page == "Overview":
    st.header("Welcome to the Wine Quality Analysis App")
    st.write(
        """
        This app allows you to explore the Wine dataset and apply machine learning algorithms 
        for wine classification. Start by exploring the data overview below, then switch to 
        the Machine Learning Models section to see the results of different algorithms.
        """
    )
    visualize_data()

elif page == "Machine Learning Models":
    if algorithm:
        display_results(algorithm)
