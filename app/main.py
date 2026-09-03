"""
==============================================================================
API DE PREDICCION DE C6H6_GT (BENCENO, PROXIMA HORA) - VERSION CONTENEDOR
==============================================================================
A diferencia de api_prediccion.py (pensado para desarrollo local, conectado
en vivo al servidor MLflow), esta version esta pensada para correr DENTRO
del contenedor Docker: carga el modelo, los escaladores y los nombres de
las features desde la carpeta local 'model_artifact/' (generada una sola
vez con export_model.py), sin conectarse a MLflow ni importar 'src'.

Esto hace que el contenedor sea autocontenido: no necesita red, ni acceso
al servidor MLflow, ni a SQL Server, para poder predecir.

Estructura esperada dentro del contenedor (ver Dockerfile):
    /app/
      app/main.py              <- este archivo
      model_artifact/
        modelo/                <- modelo sklearn guardado con mlflow.sklearn.save_model
        escaladores/
          escalador_X.pkl
          escalador_y.pkl
        feature_columns.json   <- {"feature_columns": [...], "target_column": "..."}
        RUN_ID.txt

Como correr localmente para probar (fuera de Docker), desde la raiz del
proyecto, habiendo corrido antes export_model.py:
    uvicorn app.main:app --reload --port 8000
==============================================================================
"""

# ---------------------------------------------------------------------------
# 1. IMPORTACION DE LIBRERIAS
# ---------------------------------------------------------------------------
import os
import json
import math
import logging
from pathlib import Path

import pandas as pd                                            # Construccion del DataFrame de entrada al modelo
import mlflow.sklearn                                             # Carga LOCAL del modelo (sin conexion a ningun servidor)
import joblib                                                       # Carga de los escaladores (StandardScaler) guardados

from fastapi import FastAPI, Request, status                          # Framework principal de la API
from fastapi.responses import JSONResponse                             # Respuestas de error controladas
from fastapi.exceptions import RequestValidationError                   # Excepcion que lanza FastAPI ante datos invalidos
from pydantic import BaseModel, create_model, field_validator             # Validacion de datos de entrada

# ---------------------------------------------------------------------------
# 2. CONFIGURACION Y LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

# Carpeta donde vive el modelo exportado. Dentro del contenedor, el
# Dockerfile copia 'model_artifact/' junto a 'app/' en el WORKDIR /app,
# por lo que la ruta relativa "model_artifact" (subiendo un nivel desde
# este archivo) resuelve correctamente tanto en Docker como corriendo
# localmente desde la raiz del proyecto.
RAIZ_APP = Path(__file__).resolve().parent.parent               # sube de app/ a la raiz (/app dentro del contenedor)
CARPETA_MODEL_ARTIFACT = RAIZ_APP / "model_artifact"

RUTA_MODELO = CARPETA_MODEL_ARTIFACT / "modelo"
RUTA_ESCALADOR_X = CARPETA_MODEL_ARTIFACT / "escaladores" / "escalador_X.pkl"
RUTA_ESCALADOR_Y = CARPETA_MODEL_ARTIFACT / "escaladores" / "escalador_y.pkl"
RUTA_FEATURE_COLUMNS = CARPETA_MODEL_ARTIFACT / "feature_columns.json"
RUTA_RUN_ID = CARPETA_MODEL_ARTIFACT / "RUN_ID.txt"

# ---------------------------------------------------------------------------
# 3. CARGA DE ARTEFACTOS LOCALES (al iniciar la API, una sola vez)
# ---------------------------------------------------------------------------
def _cargar_todo():
    """
    Carga el modelo, los escaladores y los metadatos de features/target
    desde la carpeta local model_artifact/. Si algo falta, se deja
    propagar la excepcion: la API no debe arrancar en un estado
    inconsistente (mejor que falle rapido al iniciar, no en el primer
    request de un usuario real).
    """
    if not CARPETA_MODEL_ARTIFACT.exists():
        raise RuntimeError(
            f"No se encontro la carpeta '{CARPETA_MODEL_ARTIFACT}'. "
            "Corré export_model.py antes de levantar esta API/contenedor."
        )

    metadata = json.loads(RUTA_FEATURE_COLUMNS.read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    target_column = metadata["target_column"]

    modelo = mlflow.sklearn.load_model(str(RUTA_MODELO))          # Carga 100% local, sin red
    escalador_X = joblib.load(RUTA_ESCALADOR_X)
    escalador_y = joblib.load(RUTA_ESCALADOR_Y)

    run_id_origen = RUTA_RUN_ID.read_text(encoding="utf-8").strip() if RUTA_RUN_ID.exists() else "desconocido"

    return modelo, escalador_X, escalador_y, feature_columns, target_column, run_id_origen


MODELO, ESCALADOR_X, ESCALADOR_Y, FEATURE_COLUMNS, TARGET_COLUMN, RUN_ID_ORIGEN = _cargar_todo()

logger.info(f"Modelo cargado localmente (run de origen: {RUN_ID_ORIGEN}).")
logger.info(f"Features esperadas por el modelo ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

# ---------------------------------------------------------------------------
# 4. DEFINICION DINAMICA DEL ESQUEMA DE ENTRADA (validacion de datos)
# ---------------------------------------------------------------------------
# Identico enfoque al de api_prediccion.py: se construye un campo float por
# cada feature, con validadores que rechazan NaN e infinito, inyectados via
# __validators__ (necesario en Pydantic v2 para que los validadores dinamicos
# se registren de verdad).
campos_dinamicos = {
    nombre_feature: (float, ...)
    for nombre_feature in FEATURE_COLUMNS
}


def _validar_no_nan_ni_infinito(valor: float, nombre_campo: str) -> float:
    if math.isnan(valor):
        raise ValueError(f"El campo '{nombre_campo}' no puede ser NaN.")
    if math.isinf(valor):
        raise ValueError(f"El campo '{nombre_campo}' no puede ser infinito.")
    return valor


validadores_dinamicos = {}
for nombre_feature in FEATURE_COLUMNS:
    def _crear_validador(nombre_campo):
        def _validador(cls, valor):
            return _validar_no_nan_ni_infinito(valor, nombre_campo)
        return _validador

    validadores_dinamicos[f"_validar_{nombre_feature}"] = field_validator(nombre_feature)(
        classmethod(_crear_validador(nombre_feature))
    )

EntradaPrediccion: type[BaseModel] = create_model(
    "EntradaPrediccion",
    __validators__=validadores_dinamicos,
    **campos_dinamicos,
)


class SalidaPrediccion(BaseModel):
    """Esquema de la respuesta exitosa del endpoint /predict."""
    target_next_hour_predicho: float
    run_id_modelo: str
    features_utilizadas: list[str]


# ---------------------------------------------------------------------------
# 5. APLICACION FASTAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de Prediccion C6H6_GT (proxima hora) - Contenedor",
    description=(
        "Sirve el modelo de Red Neuronal (MLPRegressor) exportado localmente "
        "con export_model.py. No depende de conexion a MLflow ni a SQL Server."
    ),
    version="1.0.0",
)


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


@app.exception_handler(Exception)
async def manejador_errores_generico(request: Request, exc: Exception):
    logger.exception("Error inesperado al procesar la solicitud.")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Ocurrio un error inesperado al procesar la solicitud."},
    )


@app.get("/health")
def salud():
    """Chequeo simple de que la API esta viva y el modelo quedo cargado."""
    return {
        "estado": "ok",
        "run_id_modelo": RUN_ID_ORIGEN,
        "cantidad_features": len(FEATURE_COLUMNS),
    }


@app.post("/predict", response_model=SalidaPrediccion)
def predecir(entrada: EntradaPrediccion):
    """
    Recibe los valores de las features (una fila) y devuelve la prediccion
    de target_next_hour, usando el modelo y los escaladores cargados
    localmente desde model_artifact/ (sin ninguna llamada de red).
    """
    datos_dict = entrada.model_dump()
    fila = pd.DataFrame([[datos_dict[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    fila_escalada = ESCALADOR_X.transform(fila)
    prediccion_escalada = MODELO.predict(fila_escalada)
    prediccion_real = ESCALADOR_Y.inverse_transform(
        prediccion_escalada.reshape(-1, 1)
    ).ravel()[0]

    if math.isnan(prediccion_real) or math.isinf(prediccion_real):
        logger.error(f"El modelo devolvio un valor invalido: {prediccion_real}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "El modelo genero una prediccion invalida (NaN/infinito)."},
        )

    return SalidaPrediccion(
        target_next_hour_predicho=float(prediccion_real),
        run_id_modelo=RUN_ID_ORIGEN,
        features_utilizadas=FEATURE_COLUMNS,
    )
