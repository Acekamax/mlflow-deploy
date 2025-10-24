import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ===== Parámetro de calidad =====
THRESHOLD_MSE = 3200.0  # ajusta si lo deseas

# ===== Config MLflow coherente con train.py =====
# Usa ENV en CI; si no existe, usa 'mlruns' local con URI válida (file:///...)
tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    (Path.cwd() / "mlruns").resolve().as_uri()
)
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "CI-CD-Lab2")

mlflow.set_tracking_uri(tracking_uri)
print(f"--- Debug validate: tracking_uri = {tracking_uri} ---")
print(f"--- Debug validate: experiment = {experiment_name} ---")

# Obtén el experimento
exp = mlflow.get_experiment_by_name(experiment_name)
if exp is None:
    print(f"ERROR: No existe el experimento '{experiment_name}'. Corre primero train.py.")
    sys.exit(1)

# Busca el último run FINISHED
client = MlflowClient()
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
if not runs:
    print("ERROR: No hay runs finalizados para validar.")
    sys.exit(1)

run = runs[0]
run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"
print(f"--- Debug validate: usando run_id = {run_id} ---")
print(f"--- Debug validate: model_uri = {model_uri} ---")

# Cargar modelo desde MLflow
model = mlflow.sklearn.load_model(model_uri)

# Cargar datos y replicar el split del entrenamiento
X, y = load_diabetes(return_X_y=True)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"--- Debug validate: X_test shape = {X_test.shape} ---")

# Métrica
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"🔍 MSE validación: {mse:.4f} (umbral: {THRESHOLD_MSE})")

# Criterio de aceptación
if mse <= THRESHOLD_MSE:
    print("✅ El modelo cumple los criterios de calidad.")
    sys.exit(0)
else:
    print("❌ El modelo NO cumple el umbral. Deteniendo pipeline.")
    sys.exit(1)
