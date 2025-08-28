import mlflow

# Experiment set karna
mlflow.set_experiment("iris_classification")

with mlflow.start_run():
    # Parameters log karna
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_estimators", 100)
    
    # Metrics log karna
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("loss", 0.15)
    
    # Example file/artifact save karna
    with open("output.txt", "w") as f:
        f.write("This is a sample output file.")
    mlflow.log_artifact("output.txt")
