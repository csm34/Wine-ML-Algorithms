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

# Page configuration
st.set_page_config(
    page_title="Wine Quality Analysis",
    page_icon="🍇",
    layout="wide"
)

# Title and introduction
st.title("🍇 Wine Quality Analysis Dashboard")
st.markdown("""
This dashboard analyzes wine quality data using different machine learning models.
The dataset includes various wine attributes and their classifications.
""")

# Load and prepare data
@st.cache_data
def load_data():
    wine_data = load_wine()
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    df['class'] = wine_data.target
    return df, wine_data

df, wine_data = load_data()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Analysis", "Model Training", "Model Comparison"])

# Data Overview Page
if page == "Data Overview":
    st.header("Dataset Overview")

    # Display metrics in cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Records",
            value=f"{len(df):,}"
        )

    with col2:
        st.metric(
            label="Features",
            value=len(df.columns) - 1
        )

    with col3:
        st.metric(
            label="Target Classes",
            value=len(df['class'].unique())
        )

    with col4:
        st.metric(
            label="Missing Values",
            value=df.isnull().sum().sum()
        )

    st.write("")

    # Sample Data
    st.subheader("Sample Data")
    st.dataframe(
        df.head(),
        use_container_width=True,
        height=230
    )

    # Target Class Distribution
    st.subheader("Target Class Distribution")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x='class', palette='rocket')
        plt.title('Distribution of Wine Classes')
        st.pyplot(fig)

    with col2:
        st.write("")
        st.write("")
        class_distribution = df['class'].value_counts()
        for class_name, count in class_distribution.items():
            st.metric(
                label=f"Class {class_name}",
                value=count
            )

# Exploratory Analysis Page
elif page == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")

    # Feature Distribution
    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select Feature", df.columns[:-1])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=feature_to_plot, kde=True, color='purple')
    plt.title(f'Distribution of {feature_to_plot}')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    st.pyplot(fig)

# Model Training Page
elif page == "Model Training":
    st.header("Model Training and Evaluation")

    # Data preprocessing
    X = df.drop('class', axis=1)
    y = df['class']

    # Train-test split
    test_size = st.slider("Select Test Size", 0.1, 0.4, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model selection
    model_choice = st.selectbox(
        "Select Model",
        ["KNN", "Random Forest", "LightGBM"]
    )

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            if model_choice == "KNN":
                model = KNeighborsClassifier(n_neighbors=5)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = LGBMClassifier(n_estimators=100, random_state=42)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model Performance")
                accuracy = accuracy_score(y_test, y_pred)
                st.metric(label="Accuracy", value=f"{accuracy:.4f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

            with col2:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    confusion_matrix(y_test, y_pred),
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=wine_data.target_names,
                    yticklabels=wine_data.target_names
                )
                plt.title(f'{model_choice} Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

            # Feature importance for applicable models
            if model_choice in ["Random Forest", "LightGBM"]:
                st.subheader("Feature Importance")
                feature_importance = pd.Series(
                    model.feature_importances_, index=wine_data.feature_names
                ).sort_values(ascending=False)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=feature_importance.reset_index(),
                    x=0,
                    y='index',
                    palette='viridis'
                )
                plt.title('Top Features by Importance')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                st.pyplot(fig)

# Model Comparison Page
else:
    st.header("Model Comparison")

    if st.button("Compare All Models"):
        with st.spinner("Training all models..."):
            # Data preprocessing
            X = df.drop('class', axis=1)
            y = df['class']

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train all models
            models = {
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "LightGBM": LGBMClassifier(n_estimators=100, random_state=42)
            }

            results = {}
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'predictions': y_pred
                }

            # Display comparison results
            st.subheader("Accuracy Comparison")
            accuracy_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[model]['accuracy'] for model in results.keys()]
            })

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(accuracy_df)

            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=accuracy_df,
                    x='Model',
                    y='Accuracy',
                    palette='rocket'
                )
                plt.title('Model Accuracy Comparison')
                plt.ylim(0, 1)
                st.pyplot(fig)

            # Detailed model comparison
            st.subheader("Detailed Model Performance")
            for name in results.keys():
                st.write(f"\n{name}:")
                st.text(classification_report(y_test, results[name]['predictions']))

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    confusion_matrix(y_test, results[name]['predictions']),
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=wine_data.target_names,
                    yticklabels=wine_data.target_names
                )
                plt.title(f'{name} Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

                # Feature importance for applicable models
                if name in ["Random Forest", "LightGBM"]:
                    st.subheader(f"{name} Feature Importance")
                    feature_importance = pd.Series(
                        models[name].feature_importances_, index=wine_data.feature_names
                    ).sort_values(ascending=False)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(
                        data=feature_importance.reset_index(),
                        x=0,
                        y='index',
                        palette='viridis'
                    )
                    plt.title(f'{name} Feature Importance')
                    plt.xlabel('Importance')
                    plt.ylabel('Feature')
                    st.pyplot(fig)
# Footer
st.markdown("""
---
Created with ❤️ using Streamlit
""")
