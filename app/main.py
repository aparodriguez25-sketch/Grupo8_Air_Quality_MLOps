"""
API de inferencia — predicción de C6H6_GT (benceno, próxima hora), servida
con MLflow (pyfunc) + FastAPI. Adaptado del patrón de carga robusta usado
en el proyecto de referencia (clasificación de vinos), pero para regresión
y con un esquema de entrada generado dinámicamente a partir de las features
reales del proyecto (no hay rangos físicos conocidos de antemano, como sí
los hay para la química del vino, así que se valida NaN/infinito en vez de
rangos gt/lt fijos por campo).

Endpoints:
    GET  /health   -> verifica que el servicio y el modelo están cargados
    POST /predict  -> recibe las features, devuelve la predicción de
                       target_next_hour (concentración de C6H6_GT en t+1h)

Todo se carga desde la carpeta local 'model_artifact/' (generada una sola
vez con export_model.py): no hay conexión a MLflow en vivo ni a SQL Server,
por lo que el contenedor es autocontenido.
"""

import json
import math
import logging
from pathlib import Path

import joblib
import pandas as pd
import mlflow.pyfunc                                              # pyfunc: interfaz generica de MLflow para predecir
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, create_model, field_validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

# Carpeta con el modelo y los escaladores exportados. Este archivo vive en
# app/main.py, por lo que 'model_artifact' esta un nivel arriba (la raiz del
# proyecto, tanto localmente como dentro del contenedor Docker en /app).
MODEL_DIR = Path(__file__).resolve().parent.parent / "model_artifact"

app = FastAPI(
    title="API de inferencia — Predicción de C6H6_GT (próxima hora)",
    description=(
        "Sirve un MLPRegressor entrenado y registrado en MLflow, exportado "
        "localmente. Predice la concentración de benceno una hora hacia el "
        "futuro a partir de las features del proyecto."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CARGA DEL MODELO Y ARTEFACTOS AUXILIARES (una sola vez, al iniciar la API)
# ---------------------------------------------------------------------------
# Igual que en el proyecto de referencia: si algo falla acá, el contenedor
# NO se cae. El error queda guardado en load_error, y /health lo reporta con
# 503 en vez de que el proceso entero muera al arrancar. Esto es mas facil
# de diagnosticar en produccion (el contenedor sigue vivo y respondiendo)
# que un CrashLoopBackOff sin mensaje claro.
try:
    modelo = mlflow.pyfunc.load_model(str(MODEL_DIR / "modelo"))       # Modelo cargado via interfaz generica pyfunc
    escalador_X = joblib.load(MODEL_DIR / "escaladores" / "escalador_X.pkl")
    escalador_y = joblib.load(MODEL_DIR / "escaladores" / "escalador_y.pkl")

    metadata = json.loads((MODEL_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    FEATURE_COLUMNS = metadata["feature_columns"]
    TARGET_COLUMN = metadata["target_column"]

    ruta_run_id = MODEL_DIR / "RUN_ID.txt"
    MODEL_VERSION = ruta_run_id.read_text(encoding="utf-8").strip() if ruta_run_id.exists() else "desconocida"

    load_error = None
    logger.info(f"Modelo cargado correctamente (run de origen: {MODEL_VERSION}).")
    logger.info(f"Features esperadas ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
except Exception as e:
    modelo = None
    escalador_X = None
    escalador_y = None
    FEATURE_COLUMNS = []
    TARGET_COLUMN = None
    MODEL_VERSION = "desconocida"
    load_error = str(e)
    logger.exception("No se pudo cargar el modelo o sus artefactos auxiliares.")


# ---------------------------------------------------------------------------
# ESQUEMA DE ENTRADA GENERADO DINAMICAMENTE (un campo float por feature)
# ---------------------------------------------------------------------------
# A diferencia del ejemplo de vinos (que conoce de antemano rangos fisicos
# validos por variable, como alcohol entre 0 y 20), las features de este
# proyecto (lags, medias moviles, hora/dia/mes, etc.) no tienen un rango
# universal conocido de antemano, asi que en vez de gt/lt fijos se valida
# que ningun valor sea NaN o infinito, que es lo que realmente rompe la
# prediccion sin dar un error claro.
def _validar_no_nan_ni_infinito(valor: float, nombre_campo: str) -> float:
    if math.isnan(valor):
        raise ValueError(f"El campo '{nombre_campo}' no puede ser NaN.")
    if math.isinf(valor):
        raise ValueError(f"El campo '{nombre_campo}' no puede ser infinito.")
    return valor


def _construir_esquema_entrada(feature_columns: list[str]) -> type[BaseModel]:
    """Genera el modelo Pydantic de entrada a partir de la lista de features."""
    campos = {nombre: (float, ...) for nombre in feature_columns}

    validadores = {}
    for nombre in feature_columns:
        def _crear_validador(nombre_campo):
            def _validador(cls, valor):
                return _validar_no_nan_ni_infinito(valor, nombre_campo)
            return _validador

        validadores[f"_validar_{nombre}"] = field_validator(nombre)(
            classmethod(_crear_validador(nombre))
        )

    return create_model("EntradaPrediccion", __validators__=validadores, **campos)


# Si el modelo no cargo, FEATURE_COLUMNS queda vacio: se genera un esquema
# vacio (sin campos) solo para que la app arranque; /predict va a fallar con
# 503 antes de siquiera validar, asi que esto nunca se usa de verdad en ese caso.
EntradaPrediccion = _construir_esquema_entrada(FEATURE_COLUMNS)


class PredictionResponse(BaseModel):
    target_next_hour_predicho: float
    model_version: str


# ---------------------------------------------------------------------------
# MANEJO DE ERRORES DE VALIDACION (respuestas 422 claras y legibles)
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def manejador_errores_validacion(request: Request, exc: RequestValidationError):
    errores_legibles = [
        {
            "campo": " -> ".join(str(parte) for parte in error["loc"] if parte != "body"),
            "motivo": error["msg"],
        }
        for error in exc.errors()
    ]
    logger.warning(f"Solicitud invalida recibida: {errores_legibles}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Datos de entrada invalidos.", "detalle": errores_legibles},
    )


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    if modelo is None:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {load_error}")
    return {"status": "ok", "model_version": MODEL_VERSION, "cantidad_features": len(FEATURE_COLUMNS)}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: EntradaPrediccion):
    if modelo is None:
        raise HTTPException(status_code=503, detail=f"El modelo no se cargó correctamente: {load_error}")

    # Se arma un DataFrame de una sola fila, respetando el MISMO orden de
    # columnas con el que se entreno el modelo (StandardScaler es sensible
    # al orden de las columnas).
    datos_dict = features.model_dump()
    fila = pd.DataFrame([[datos_dict[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    # Se escala la entrada con el mismo escalador de entrenamiento
    fila_escalada = pd.DataFrame(escalador_X.transform(fila), columns=FEATURE_COLUMNS)

    # Prediccion via la interfaz generica pyfunc (funciona igual sin importar
    # que flavor de sklearn se haya usado para loguear el modelo original)
    prediccion_escalada = modelo.predict(fila_escalada)
    valor_escalado = float(pd.Series(prediccion_escalada).iloc[0])

    # Se revierte el escalado para volver a las unidades reales de C6H6_GT
    prediccion_real = escalador_y.inverse_transform([[valor_escalado]])[0][0]

    if math.isnan(prediccion_real) or math.isinf(prediccion_real):
        logger.error(f"El modelo devolvio un valor invalido: {prediccion_real}")
        raise HTTPException(status_code=500, detail="El modelo genero una prediccion invalida (NaN/infinito).")

    return {
        "target_next_hour_predicho": float(prediccion_real),
        "model_version": MODEL_VERSION,
    }
