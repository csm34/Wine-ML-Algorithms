# Wine Classification with Machine Learning

This machine learning project uses the Wine dataset for classification with three algorithms: KNN, Random Forest, and LightGBM. The repository contains a Jupyter Notebook for data analysis and model training, along with a Streamlit app for interactive results exploration.

---

## Project Overview

In this project, we explore the Wine dataset to classify wines into different categories based on various features such as alcohol content, color intensity, and more. The classification is performed using three machine learning algorithms:

- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **LightGBM**

The project is divided into two main parts:
1. **Jupyter Notebook**: Used for data exploration, preprocessing, model training, and evaluation.
2. **Streamlit App**: Provides an interactive interface to visualize the results and select different machine learning models for predictions.

---

## Files in the Repository

- **`wine.ipynb`**: Jupyter Notebook containing the data analysis and machine learning model training process.
- **`streamlit_wine_app.py`**: Streamlit application for interactive results exploration.
- **`requirements.txt`**: List of dependencies required to run the project.

---

## Required Dependencies

This project requires the following Python libraries:

- `pandas` for data manipulation
- `numpy` for numerical computations
- `matplotlib` for data visualization
- `seaborn` for statistical data visualization
- `scikit-learn` for machine learning models and metrics
- `lightgbm` for the LightGBM algorithm
- `streamlit` for creating the interactive web app

---

## Setup and Installation

1. Clone this repository to your local machine.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   ```
3. Install the required dependencies
4. Run the Streamlit app:
   ```bash
   streamlit run wine_app.py
   ```
---

## Usage
1. Upon launching the Streamlit app, the main page will display information about the Wine dataset.
2. From there, you can choose one of the machine learning algorithms (KNN, Random Forest, or LightGBM) to see the results of the classification model.

---
## Screenshots

### Main Page
![Wine Dataset](images/wine_dataset.png)

### Machine Learning Model Selection
![Wine ML Selection](images/wine_ml.png)

---

## Licence
This project is open-source and available under the MIT License.

