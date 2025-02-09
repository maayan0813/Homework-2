import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import csv
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from flaml import AutoML

import h2o
from h2o.automl import H2OAutoML
import xgboost
import lightgbm

import matplotlib
matplotlib.use("Agg") 

# Load the dataset
def load_data():
    print("Loading data...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    sample_submission_df = pd.read_csv("sample_submission.csv")
    print("Data loaded successfully.")
    return train_df, test_df, sample_submission_df

train_df, test_df, sample_submission_df = load_data()

# Data preprocessing
def preprocess_data(train_df, test_df):
    print("Preprocessing data...")
    X = train_df.drop(columns=["id", "smoking"])
    y = train_df["smoking"]
    
    imputer = SimpleImputer(strategy="median") 
    X_imputed = imputer.fit_transform(X)
    test_imputed = imputer.transform(test_df.drop(columns=["id"]))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    test_scaled = scaler.transform(test_imputed)
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("Data preprocessing completed.")
    return X_train, X_val, y_train, y_val, test_scaled, test_df

X_train, X_val, y_train, y_val, test_scaled, test_df = preprocess_data(train_df, test_df)

# Train Random Forest model
def train_random_forest(X_train, y_train, X_val, y_val):
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    y_val_pred = rf_model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_val_pred)
    print(f"Random Forest trained. AUC: {auc_score:.4f}")
    return rf_model, y_val_pred, auc_score

rf_model, y_rf_pred, auc_rf = train_random_forest(X_train, y_train, X_val, y_val)

# Train FLAML AutoML Model
def train_flaml(X_train, y_train, X_val, y_val):
    print("Training FLAML AutoML...")
    automl = AutoML()
    automl.fit(X_train, y_train, task="classification", metric="roc_auc", time_budget=600)  
    y_val_pred_flaml = automl.predict_proba(X_val)[:, 1]
    auc_flaml = roc_auc_score(y_val, y_val_pred_flaml)
    print(f"FLAML training completed. AUC: {auc_flaml:.4f}")
    return automl, y_val_pred_flaml, auc_flaml

automl, y_flaml_pred, auc_flaml = train_flaml(X_train, y_train, X_val, y_val)

# Train H2O AutoML Model
def train_h2o(train_df):
    print("Training H2O AutoML...")
    h2o.init()
    train_h2o = h2o.H2OFrame(train_df)
    train_h2o["smoking"] = train_h2o["smoking"].asfactor()  
    aml = H2OAutoML(max_models=10, seed=42)
    aml.train(x=train_h2o.columns[:-1], y="smoking", training_frame=train_h2o)
    print("H2O AutoML training completed.")
    return aml

aml_h2o = train_h2o(train_df)

# Generate and save AUC curves
def plot_auc_curve(y_true, y_preds, model_name, filename):
    fpr, tpr, _ = roc_curve(y_true, y_preds)
    auc_score = roc_auc_score(y_true, y_preds)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color='blue')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(filename)
    print(f"AUC curve saved as {filename}")

# Save AUC curves for each model
plot_auc_curve(y_val, y_rf_pred, "Random Forest", "auc_curve_random_forest.png")
plot_auc_curve(y_val, y_flaml_pred, "FLAML AutoML", "auc_curve_flaml.png")

# Save results to log file
print("Saving results to log file...")
run_data = ["FLAML AutoML", "Time Budget = 600 sec", auc_flaml]
with open("runs_log.csv", "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(run_data)
print("Results saved.")


def generate_submission(automl, test_scaled, test_df, filename="submission.csv"):
    print("Generating submission file...")
    test_predictions = automl.predict_proba(test_scaled)[:, 1]
    submission = pd.DataFrame({"id": test_df["id"], "smoking": test_predictions})
    submission.to_csv(filename, index=False)
    print(f"Submission file saved as {filename}")

generate_submission(automl, test_scaled, test_df)

# Explain the model with SHAP
def explain_model(automl, X_val):
    print("Running SHAP analysis...")
    best_model = automl.model.estimator
    
    if isinstance(best_model, (xgboost.XGBClassifier, lightgbm.LGBMClassifier)):
        explainer = shap.Explainer(best_model)
        shap_values = explainer(X_val)

        # Explain 3 samples
        for i in range(3):
            shap.plots.waterfall(shap_values[i], show=False)
            plt.savefig(f"shap_explanation_{i}.png")

        # Feature importance plot
        shap.summary_plot(shap_values, X_val, show=False)
        plt.savefig("shap_summary.png")

        # SHAP Dependence Contribution for Top 5 Features
        top_features = np.argsort(np.abs(shap_values.values).mean(0))[-5:]
        for feature in top_features:
            shap.dependence_plot(feature, shap_values, X_val, show=False)
            plt.savefig(f"shap_dependence_{feature}.png")

        print("SHAP explanations saved.")
    else:
        print("SHAP does not support this model directly.")

explain_model(automl, X_val)
