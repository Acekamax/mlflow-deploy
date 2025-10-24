import os
import sys
import traceback
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print(f"--- Debug: Initial CWD: {Path.cwd()} ---")

# ========= Config MLflow (portátil) =========
# 1) Nombre del experimento: toma ENV si existe (CI), si no usa default (local)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "CI-CD-Lab2")

# 2) Tracking URI:
#    - Si hay ENV (CI), úsalo tal cual (p. ej. file://${{ github.workspace }}/mlruns)
#    - Si no, usa carpeta local 'mlruns' con URI válida (file:///C:/... en Windows)
tracking_uri_env = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri_env:
    tracking_uri = tracking_uri_env
else:
    mlruns_dir = (Path.cwd() / "mlruns")
    mlruns_dir.mkdir(exist_ok=True)
    tracking_uri = mlruns_dir.resolve().as_uri()

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Experiment name: {EXPERIMENT_NAME} ---")

# ========= Entrenamiento =========
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression().fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"--- Debug: MSE train/test = {mse:.4f} ---")

# ========= MLflow Run + artefactos =========
run = None
try:
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: Artifact URI del run: {run.info.artifact_uri} ---")

        # params/metrics
        mlflow.log_params({"model": "LinearRegression", "test_size": 0.2, "random_state": 42})
        mlflow.log_metric("mse", mse)

        # firma e input_example para evitar warning
        signature = infer_signature(X_train, model.predict(X_train[:5]))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test[:5]
        )

        # Guardar además model.pkl en la raíz (útil para descargar rápido)
        pkl_path = Path.cwd() / "model.pkl"
        joblib.dump(model, pkl_path)
        mlflow.log_artifact(str(pkl_path))
        print(f"✅ Modelo logueado en MLflow y guardado como {pkl_path}. MSE: {mse:.4f}")

except Exception:
    print("\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    try:
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        if run:
            print(f"Artifact URI del run: {run.info.artifact_uri}")
    except Exception:
        pass
    sys.exit(1)
