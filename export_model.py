"""
Exporta un modelo (y sus escaladores) desde un RUN de MLflow a una carpeta
local autocontenida, lista para copiar dentro de la imagen Docker.

A diferencia de la version anterior, este script NO depende del Model
Registry (no necesita un modelo registrado ni un alias como "production").
Apunta directamente a un run_id concreto usando el esquema runs:/, que es
el mismo mecanismo que ya usa api_prediccion.py.

Correr UNA VEZ, en tu entorno local, con el mlflow server ya levantado
(mismo tracking URI que usaste para entrenar/registrar el modelo).

Uso:
    python export_model.py
"""

import sys
import os
import json
import shutil
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

# Este script vive en la raiz del proyecto (junto a 'src'), asi que alcanza
# con un solo dirname. Se agrega igual por consistencia con entrenamiento2.py
# y api_prediccion.py, y por si en el futuro se mueve de carpeta.
RAIZ_PROYECTO = os.path.dirname(os.path.abspath(__file__))
sys.path.append(RAIZ_PROYECTO)

# Se importan las columnas oficiales de features/target, para exportarlas
# junto con el modelo. Esto le permite al contenedor Docker (app/main.py)
# construir el esquema de entrada de la API SIN necesitar importar 'src'
# dentro de la imagen (que ademas arrastraria dependencias de ingestion,
# como pyodbc/sqlalchemy, innecesarias en un contenedor de solo inferencia).
from src.features.transformations import FEATURE_COLUMNS, MODEL_TARGET_COLUMN

# --- Ajusta estos valores a tu proyecto ---
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "prediccion_C6H6_GT_red_neuronal"
OUTPUT_DIR = "model_artifact"

# Si ya sabes el run_id exacto que queres exportar, pegalo aca entre
# comillas (por ejemplo "7372fd65f95c4026a145ad8b52dba50b"). Si lo dejas
# vacio (""), el script busca automaticamente, entre todas las corridas del
# experimento, la de menor rmse_test que ademas cargue sin errores (mismo
# criterio que usa api_prediccion.py para evitar corridas incompletas).
RUN_ID = ""
# ---------------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)
cliente = mlflow.tracking.MlflowClient()


def _intentar_cargar(run_id: str):
    """
    Intenta cargar el modelo y los dos escaladores de un run puntual.
    Devuelve (modelo, escalador_X, escalador_y) si todo carga bien, o
    lanza una excepcion si falta algo (corrida incompleta/interrumpida).
    """
    ruta_escalador_X = cliente.download_artifacts(run_id, "escaladores/escalador_X.pkl")
    ruta_escalador_y = cliente.download_artifacts(run_id, "escaladores/escalador_y.pkl")

    escalador_X = joblib.load(ruta_escalador_X)
    escalador_y = joblib.load(ruta_escalador_y)

    modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo_red_neuronal")

    return modelo, escalador_X, escalador_y


def _seleccionar_run_automaticamente() -> str:
    """
    Busca, entre las corridas del experimento ordenadas por rmse_test
    ascendente, la primera que cargue sin errores (modelo + escaladores).
    """
    experimento = cliente.get_experiment_by_name(EXPERIMENT_NAME)
    if experimento is None:
        raise RuntimeError(
            f"No se encontro el experimento '{EXPERIMENT_NAME}' en {TRACKING_URI}."
        )

    corridas = cliente.search_runs(
        experiment_ids=[experimento.experiment_id],
        order_by=["metrics.rmse_test ASC"],
        max_results=200,
    )

    for run in corridas:
        rmse_test = run.data.metrics.get("rmse_test")
        if rmse_test is None:
            continue
        try:
            _intentar_cargar(run.info.run_id)               # Solo se prueba que cargue
            print(f"Run seleccionado automaticamente: {run.info.run_id} (rmse_test={rmse_test})")
            return run.info.run_id
        except Exception as error:
            print(f"  Run {run.info.run_id} descartado (incompleto): {error}")

    raise RuntimeError(
        f"Ninguna corrida de '{EXPERIMENT_NAME}' pudo cargarse completamente. "
        "Revisa que entrenamiento2.py se haya ejecutado sin interrupciones."
    )


# ---------------------------------------------------------------------------
# 1. Determinar que run_id se va a exportar
# ---------------------------------------------------------------------------
run_id_a_exportar = RUN_ID if RUN_ID else _seleccionar_run_automaticamente()

# ---------------------------------------------------------------------------
# 2. Preparar la carpeta de salida (se recrea vacia)
# ---------------------------------------------------------------------------
output_path = Path(OUTPUT_DIR)
if output_path.exists():
    shutil.rmtree(output_path)                                  # save_model falla si la carpeta ya existe

# ---------------------------------------------------------------------------
# 3. Descargar y guardar el modelo, en formato sklearn autocontenido
# ---------------------------------------------------------------------------
model_uri = f"runs:/{run_id_a_exportar}/modelo_red_neuronal"
print(f"Descargando modelo desde: {model_uri}")
modelo = mlflow.sklearn.load_model(model_uri)

print(f"Guardando modelo autocontenido en: {OUTPUT_DIR}/modelo/")
# Se fuerza serialization_format="pickle" en vez del formato "skops" (default
# en MLflow 3.x), que bloquea por seguridad tipos internos de sklearn como el
# optimizador Adam del MLPRegressor. "pickle" es el formato clasico, sin esa
# restriccion, y es el mismo que ya se uso al loguear el modelo originalmente
# en entrenamiento2.py (log_model con serialization_format="pickle").
mlflow.sklearn.save_model(
    modelo,
    str(output_path / "modelo"),
    serialization_format="pickle",
)

# ---------------------------------------------------------------------------
# 4. Descargar y copiar tambien los escaladores (el modelo solo no alcanza)
# ---------------------------------------------------------------------------
print("Descargando escaladores...")
ruta_escalador_X = cliente.download_artifacts(run_id_a_exportar, "escaladores/escalador_X.pkl")
ruta_escalador_y = cliente.download_artifacts(run_id_a_exportar, "escaladores/escalador_y.pkl")

carpeta_escaladores = output_path / "escaladores"
carpeta_escaladores.mkdir(parents=True, exist_ok=True)
shutil.copy(ruta_escalador_X, carpeta_escaladores / "escalador_X.pkl")
shutil.copy(ruta_escalador_y, carpeta_escaladores / "escalador_y.pkl")

print(f"Escaladores copiados a: {OUTPUT_DIR}/escaladores/")

# ---------------------------------------------------------------------------
# 5. Guardar las columnas de features/target en un JSON local
# ---------------------------------------------------------------------------
# Esto es lo que le permite al contenedor Docker armar el esquema de entrada
# de la API sin tener que importar src.features.transformations. Si el
# equipo agrega/quita una feature y vuelve a correr este script antes de
# reconstruir la imagen, el contenedor queda automaticamente sincronizado.
metadata = {
    "feature_columns": FEATURE_COLUMNS,
    "target_column": MODEL_TARGET_COLUMN,
}
(output_path / "feature_columns.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"Columnas de features/target guardadas en: {OUTPUT_DIR}/feature_columns.json")

# ---------------------------------------------------------------------------
# 6. Guardar de que run_id salio todo esto, para trazabilidad
# ---------------------------------------------------------------------------
(output_path / "RUN_ID.txt").write_text(run_id_a_exportar, encoding="utf-8")

print("\nListo. La carpeta", OUTPUT_DIR, "contiene:")
print("  - modelo/               (modelo sklearn autocontenido)")
print("  - escaladores/           (escalador_X.pkl y escalador_y.pkl)")
print("  - feature_columns.json   (nombres de features y target, sin depender de src)")
print("  - RUN_ID.txt             (de que corrida de MLflow proviene)")
print("\nCopia esta carpeta dentro del proyecto Docker (junto al Dockerfile).")
print("El contenedor ya NO necesitara conectarse a tu mlflow server para predecir.")
