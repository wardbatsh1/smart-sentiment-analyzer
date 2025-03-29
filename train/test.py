# test_mlflow_dagshub.py
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/ward.batsh2/smart-sentiment-analyzer.mlflow")
mlflow.set_experiment("sentiment-analysis")

with mlflow.start_run(run_name="test-run"):
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.95)
