import mlflow.sklearn

# Replace <RUN_ID> with the actual run ID you got from MLflow UI
RUN_ID = "<your-best-run-id>"

# Load the saved model from MLflow
model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")

# Example input (Iris dataset: sepal_length, sepal_width, petal_length, petal_width)
sample = [[5.1, 3.5, 1.4, 0.2]]

pred = model.predict(sample)
print("Prediction for sample input:", pred)
