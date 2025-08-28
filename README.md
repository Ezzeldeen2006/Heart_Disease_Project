# Heart Disease Prediction Project

## Overview

This project predicts the likelihood of heart disease using machine learning models trained on a heart disease dataset. It includes data preprocessing, feature engineering, model training, hyperparameter tuning, and a Streamlit web application for easy interaction.

---

## Project Structure

```
Heart_Disease_Project/
│── data/                  # CSV files
│── notebooks/             # Jupyter notebooks for each step
│── models/                # Trained ML models (.pkl)
│── ui/                    # Streamlit app (app.py)
│── deployment/            # Ngrok setup instructions
│── results/               # Evaluation metrics
│── requirements.txt       # Python dependencies
│── README.md              # Project overview
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Ezzeldeen2006/Heart_Disease_Project.git
cd Heart_Disease_Project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
cd ui
streamlit run app.py
```

* The public Ngrok URL will automatically appear in the sidebar:

```
Public app URL: https://abcd-1234.ngrok-free.app
```

---

## Notebooks

* `01_data_preprocessing.ipynb` – Data cleaning and preprocessing
* `02_pca_analysis.ipynb` – Principal Component Analysis
* `03_feature_selection.ipynb` – Feature selection methods
* `04_supervised_learning.ipynb` – Supervised model training
* `05_unsupervised_learning.ipynb` – Clustering and unsupervised analysis
* `06_hyperparameter_tuning.ipynb` – Model optimization

---

## Model

* `models/final_model.pkl` – Trained final model

---

## Deployment Notes

* Ngrok is used to create a public URL for the Streamlit app.
* For manual Ngrok instructions, see `deployment/ngrok_setup.txt`.
* If you get an ngrok session error (`ERR_NGROK_108`), terminate any previous ngrok processes and restart the app.
