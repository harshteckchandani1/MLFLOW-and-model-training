import argparse
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

def train_model(model_type, params):
    # Load dataset (Iris)
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "logistic":
        model = LogisticRegression(C=params["C"], max_iter=200)
    elif model_type == "randomforest":
        model = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"])
    else:
        raise ValueError("Unsupported model type")

    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("accuracy", acc)

        # Log confusion matrix as image
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title(f"Confusion Matrix ({model_type})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # Save model
        mlflow.sklearn.log_model(model, "model")

    print(f"{model_type} trained with accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="logistic or randomforest")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization strength (for logistic)")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees (for randomforest)")
    parser.add_argument("--max_depth", type=int, default=5, help="Tree depth (for randomforest)")
    args = parser.parse_args()

    if args.model == "logistic":
        params = {"C": args.C}
    else:
        params = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}

    train_model(args.model, params)
