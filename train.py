import os
import sys
import traceback
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# =========================
# Paths (compatibles Windows)
# =========================
workspace_dir = Path.cwd()                  # p.ej. C:\proyecto final mlops\mlflow-deploy
mlruns_dir = (workspace_dir / "mlruns")
mlruns_dir.mkdir(exist_ok=True)

# URI correcta tipo file:///C:/... (ojo: los espacios salen como %20 en la impresión, es normal)
tracking_uri = mlruns_dir.resolve().as_uri()
artifact_location = tracking_uri

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Desired Artifact Location Base: {artifact_location} ---")

# =========================
# Configurar MLflow
# =========================
mlflow.set_tracking_uri(tracking_uri)
experiment_name = "CI-CD-Lab2"

# Crear/obtener experimento (robusto a "ya existe")
try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location
    )
    print(f"--- Debug: Creado experimento '{experiment_name}' con ID: {experiment_id} ---")
except Exception as e:
    print(f"--- Debug: No se pudo crear (posible 'ya existe'): {e} ---")
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"--- ERROR: No se pudo obtener experimento '{experiment_name}' tras el fallo de creación. ---")
        sys.exit(1)
    experiment_id = exp.experiment_id
    print(f"--- Debug: Experimento existente ID: {experiment_id} ---")
    print(f"--- Debug: artifact_location existente: {exp.artifact_location} ---")
    if exp.artifact_location != artifact_location:
        print(f"--- WARNING: artifact_location existente ('{exp.artifact_location}') != deseado ('{artifact_location}'). ---")

# =========================
# Entrenamiento
# =========================
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

print(f"--- Debug: MSE train/test = {mse:.4f} ---")

# =========================
# Run de MLflow + guardado model.pkl
# =========================
print(f"--- Debug: Iniciando run en experimento ID: {experiment_id} ---")
run = None
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: Artifact URI del run: {artifact_uri} ---")

        # Métrica
        mlflow.log_metric("mse", mse)

        # Guardar modelo en MLflow (artefacto 'model')
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        # Guardar también un archivo local model.pkl en la raíz del proyecto
        pkl_path = workspace_dir / "model.pkl"
        joblib.dump(model, pkl_path)
        print(f"📦 Modelo guardado localmente como: {pkl_path}")

        # Subir ese model.pkl como artefacto adicional del run
        mlflow.log_artifact(str(pkl_path))
        print("✅ Modelo logueado en MLflow y model.pkl cargado como artefacto adicional.")
        print(f"✅ Entrenamiento OK. The MSE is: {mse:.4f}")

except Exception as e:
    print("\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    print("--- Fin de traza ---")
    print(f"CWD en error: {os.getcwd()}")
    try:
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment ID: {experiment_id}")
        if run:
            print(f"Artifact URI del run: {run.info.artifact_uri}")
    except Exception:
        pass
    sys.exit(1)
