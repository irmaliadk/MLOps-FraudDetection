import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def register_best_model():
    """
    Ambil run terbaik dari MLflow dan daftarkan ke Model Registry.
    Menggunakan model aliases (champion/challenger) sesuai MLflow 2.9+
    sebagai pengganti stages yang sudah deprecated.
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

    best_run = runs[0]
    run_id   = best_run.info.run_id
    run_name = best_run.info.run_name
    f1       = best_run.data.metrics["f1_score"]

    print(f"Best run    : {run_name}")
    print(f"Run ID      : {run_id}")
    print(f"F1 Score    : {f1:.4f}")

    model_uri = f"runs:/{run_id}/model"
    result    = mlflow.register_model(
        model_uri=model_uri,
        name="fraud-detection-best-model"
    )
    version = result.version
    print(f"Model registered as version {version}")

    all_versions = client.search_model_versions("name='fraud-detection-best-model'")
    for v in all_versions:
        if v.version != version:
            try:
                client.delete_registered_model_alias(
                    name="fraud-detection-best-model",
                    alias="champion"
                )
            except Exception:
                pass
            client.set_registered_model_alias(
                name="fraud-detection-best-model",
                version=v.version,
                alias="challenger"
            )

    client.set_registered_model_alias(
        name="fraud-detection-best-model",
        version=version,
        alias="champion"
    )
    print(f"Version {version} set as alias 'champion' (Production-equivalent)")

    client.update_model_version(
        name="fraud-detection-best-model",
        version=version,
        description=f"Best model from run '{run_name}' | F1 Score: {f1:.4f}"
    )

    return version

if __name__ == "__main__":
    version = register_best_model()
    print(f"\n✅ Model v{version} successfully registered as champion!")
    print("   Load via: mlflow.pyfunc.load_model('models:/fraud-detection-best-model@champion')")
