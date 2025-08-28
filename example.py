import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from mlflow.models.signature import infer_signature

# 1. Load dataset
data = load_diabetes()
X, y = data.data, data.target

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define MLflow experiment
mlflow.set_experiment("Linear_Regression_Experiment")

with mlflow.start_run():
    # 4. Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Predict
    y_pred = model.predict(X_test)

    # 6. Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = mse ** 0.5

    # 7. Log parameters and metrics
    mlflow.log_param("fit_intercept", model.fit_intercept)
    mlflow.log_param("normalize", "Not used in sklearn >=1.0")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("rmse", rmse)

    signature = infer_signature(X_train, model.predict(X_train))

    # 8. Log model
    mlflow.sklearn.log_model(
        sk_model=model, 
        name="linear_regression_model",
        signature=signature,
        input_example=X_train[:5]
        )

    print("Model logged in MLflow")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
