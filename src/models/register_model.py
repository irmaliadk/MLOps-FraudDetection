import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def register_best_model():
    """
    Ambil run terbaik dari MLflow dan daftarkan ke Model Registry
    dengan transisi stage otomatis ke Staging lalu Production.
    """
    client = MlflowClient()

    experiment = client.get_experiment_by_name("fraud-detection-experiments")
    if not experiment:
        raise ValueError("Experiment tidak ditemukan!")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("Tidak ada run yang ditemukan!")

    best_run    = runs[0]
    run_id      = best_run.info.run_id
    run_name    = best_run.info.run_name
    f1          = best_run.data.metrics["f1_score"]

    print(f"Best run    : {run_name}")
    print(f"Run ID      : {run_id}")
    print(f"F1 Score    : {f1:.4f}")

    model_uri = f"runs:/{run_id}/model"
    result    = mlflow.register_model(
        model_uri=model_uri,
        name="fraud-detection-best-model"
    )

    print(f"Model registered as version {result.version}")

    client.transition_model_version_stage(
        name="fraud-detection-best-model",
        version=result.version,
        stage="Staging"
    )
    print(f"Version {result.version} transitioned to Staging!")

    client.transition_model_version_stage(
        name="fraud-detection-best-model",
        version=result.version,
        stage="Production"
    )
    print(f"Version {result.version} transitioned to Production!")

    client.update_model_version(
        name="fraud-detection-best-model",
        version=result.version,
        description=f"Best model from run {run_name} with F1 Score {f1:.4f}"
    )

    return result.version

if __name__ == "__main__":
    version = register_best_model()
    print(f"\n✅ Model v{version} successfully registered and promoted to Production!")