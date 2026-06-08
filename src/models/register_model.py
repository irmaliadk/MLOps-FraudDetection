import mlflow
from mlflow.tracking import MlflowClient
import os

DAGSHUB_URI = "https://dagshub.com/irmaliadk/MLOps-FraudDetection.mlflow"
LOCAL_URI   = "sqlite:///mlflow.db"

if os.getenv("DAGSHUB_TOKEN"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = "irmaliadk"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri(DAGSHUB_URI)
    mlflow.set_registry_uri(DAGSHUB_URI)
    print("MLflow tracking: DagsHub")
else:
    mlflow.set_tracking_uri(LOCAL_URI)
    mlflow.set_registry_uri(LOCAL_URI)
    print("MLflow tracking: Local SQLite")

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

    # Cari logged model dari run ini
    try:
        logged_models = client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"source_run_id = '{run_id}'"
        )
        if logged_models:
            model_uri = logged_models[0].model_uri
            print(f"Using logged model URI: {model_uri}")
        else:
            model_uri = f"runs:/{run_id}/model"
            print(f"Using run artifact URI: {model_uri}")
    except Exception:
        model_uri = f"runs:/{run_id}/model"
        print(f"Using fallback URI: {model_uri}")
    
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
