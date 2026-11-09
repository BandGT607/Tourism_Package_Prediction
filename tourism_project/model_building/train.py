import pandas as pd
import xgboost as xgb
import mlflow
import joblib
import os
import warnings
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

categorical_cols = Xtrain.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = Xtrain.select_dtypes(exclude=['object', 'category']).columns.tolist()

preprocessor = make_column_transformer(
    (StandardScaler(), numerical_cols),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
)

# Force XGBoost to balance the "Yes" vs "No" predictions
ratio = float(ytrain.value_counts()[0] / ytrain.value_counts()[1])
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=ratio)

pipeline = make_pipeline(preprocessor, model)
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.01, 0.1]
}

mlflow.set_experiment("Tourism_Package_Prediction")

with mlflow.start_run():
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(Xtrain, ytrain)
    
    best_model = grid.best_estimator_
    ypred = best_model.predict(Xtest)
    
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("test_accuracy", accuracy_score(ytest, ypred))
    
    # Removed the incompatible skops_trusted_types argument
    mlflow.sklearn.log_model(best_model, "best_xgb_model")
    
    os.makedirs("tourism_project/deployment", exist_ok=True)
    joblib.dump(best_model, "tourism_project/deployment/model.joblib")
    print("Model trained and saved successfully.")