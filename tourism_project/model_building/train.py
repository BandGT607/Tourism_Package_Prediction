import pandas as pd
import xgboost as xgb
import mlflow
import joblib
import os
import sys
import warnings
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

# 1. Load the prepared data splits
try:
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()
except FileNotFoundError:
    print("ERROR: Data splits not found. Please run prep.py first.")
    sys.exit(1)

# 2. Setup Preprocessing Pipeline
categorical_cols = Xtrain.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = Xtrain.select_dtypes(exclude=['object', 'category']).columns.tolist()

preprocessor = make_column_transformer(
    (StandardScaler(), numerical_cols),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
)

# 3. Calculate Weight Ratio and Create Model Pipeline
ratio = float(ytrain.value_counts()[0] / ytrain.value_counts()[1])
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=ratio)
pipeline = make_pipeline(preprocessor, model)

# 4. Define Hyperparameter Grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.01, 0.1]
}

# 5. MLflow Experiment Tracking & Training
mlflow.set_experiment("Tourism_Package_Prediction")

with mlflow.start_run():
    print("Starting Model Training and Hyperparameter Tuning...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(Xtrain, ytrain)
    
    best_model = grid.best_estimator_
    
    # 6. Evaluation
    ypred = best_model.predict(Xtest)
    acc = accuracy_score(ytest, ypred)
    
    print("\n--- Balanced Classification Report ---")
    print(classification_report(ytest, ypred))
    
    # 7. Log metrics and params to MLflow
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("test_accuracy", acc)
    
    # ONE-SHOT FIX: Safely handle both old and new MLflow security protocols
    try:
        trusted_types = [
            "xgboost.sklearn.XGBClassifier",
            "xgboost.core.Booster",
            "sklearn.compose._column_transformer._RemainderColsList"
        ]
        mlflow.sklearn.log_model(best_model, "best_xgb_model", skops_trusted_types=trusted_types)
    except TypeError:
        # Fallback for older MLflow versions (like your local Mac)
        mlflow.sklearn.log_model(best_model, "best_xgb_model")
    
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Test Accuracy: {acc:.4f}")

    # 8. Save the model artifact for Deployment
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nModel saved successfully to {model_path}")
