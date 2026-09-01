"""
==============================================================================
API DE PREDICCION DE C6H6_GT (BENCENO, PROXIMA HORA) CON FASTAPI
==============================================================================
Objetivo:
    Exponer el mejor modelo de Red Neuronal (MLPRegressor) registrado en
    MLflow como un servicio HTTP, con validacion estricta de los datos de
    entrada (manejo de valores invalidos: tipos incorrectos, campos
    faltantes, NaN, infinitos, cuerpos vacios o mal formados).


Variables de entorno opcionales:
    MLFLOW_TRACKING_URI  -> por defecto "http://127.0.0.1:5000"
    MLFLOW_EXPERIMENT    -> por defecto "prediccion_C6H6_GT_red_neuronal"
    MLFLOW_RUN_ID        -> si se especifica, se usa esa corrida puntual en
                             vez de buscar automaticamente la de menor RMSE
==============================================================================
"""

# ---------------------------------------------------------------------------
# 0. RUTA RAIZ DEL PROYECTO (para que los imports de 'src' funcionen sin
#    importar desde que carpeta se ejecute uvicorn)
# ---------------------------------------------------------------------------
import sys
import os

# Este archivo se asume ubicado en la raiz del proyecto, junto a 'src'. Si lo
# movés a otra carpeta, ajustá esta linea igual que se hizo en entrenamiento2.py
# (por ejemplo con dirname(dirname(...)) si queda un nivel mas abajo).
# Este archivo vive en la subcarpeta 'api', un nivel debajo de la raiz del
# proyecto (donde esta 'src'). Por eso se sube UN nivel extra con un segundo
# dirname(), igual que se hizo en entrenamiento2.py.
RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RAIZ_PROYECTO)

# ---------------------------------------------------------------------------
# 1. IMPORTACION DE LIBRERIAS
# ---------------------------------------------------------------------------
import math                                                  # Para detectar NaN e infinito
import logging                                                # Para loguear errores del lado del servidor
from typing import Optional

import numpy as np                                            # Operaciones numericas
import pandas as pd                                            # Construccion del DataFrame de entrada al modelo
import mlflow                                                    # Cliente de MLflow, para buscar runs y cargar artefactos
import mlflow.sklearn                                             # Carga de modelos sklearn desde MLflow
import joblib                                                       # Carga de los escaladores (StandardScaler) guardados

from fastapi import FastAPI, Request, status                          # Framework principal de la API
from fastapi.responses import JSONResponse                             # Respuestas de error controladas
from fastapi.exceptions import RequestValidationError                   # Excepcion que lanza FastAPI ante datos invalidos
from pydantic import BaseModel, create_model, field_validator             # Validacion de datos de entrada

# Se reutiliza la lista oficial de features del proyecto, para que la API
# quede automaticamente sincronizada con lo que espera el modelo (si el
# equipo agrega/quita una feature en transformations.py, la API se adapta
# sin necesidad de tocar este archivo).
from src.features.transformations import FEATURE_COLUMNS, MODEL_TARGET_COLUMN

# ---------------------------------------------------------------------------
# 2. CONFIGURACION Y LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)                       # Configuracion basica de logs en consola
logger = logging.getLogger("api_prediccion")                    # Logger propio de este modulo

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "prediccion_C6H6_GT_red_neuronal")
MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID")                     # None si no se definio explicitamente

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)                     # Se apunta al mismo servidor usado en entrenamiento

# ---------------------------------------------------------------------------
# 3. CARGA DEL MODELO Y LOS ESCALADORES DESDE MLFLOW (al iniciar la API)
# ---------------------------------------------------------------------------
def cargar_artefactos(run_id: str):
    """
    Carga desde MLflow, para un run_id dado:
      - el modelo entrenado (MLPRegressor)
      - el escalador de features (escalador_X)
      - el escalador del target (escalador_y)
    Los escaladores se descargan como artefactos y se deserializan con joblib.
    Si algo falta o no se puede cargar, esta funcion deja propagar la
    excepcion (el llamador decide si probar con otro run o abortar).
    """
    cliente = mlflow.tracking.MlflowClient()

    # Se descargan los dos archivos .pkl de escaladores a una carpeta temporal local
    ruta_escalador_X = cliente.download_artifacts(run_id, "escaladores/escalador_X.pkl")
    ruta_escalador_y = cliente.download_artifacts(run_id, "escaladores/escalador_y.pkl")

    escalador_X = joblib.load(ruta_escalador_X)                    # Se deserializa el escalador de features
    escalador_y = joblib.load(ruta_escalador_y)                     # Se deserializa el escalador del target

    # Se carga el modelo sklearn nativo registrado en el run
    modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo_red_neuronal")

    return modelo, escalador_X, escalador_y


def seleccionar_y_cargar_modelo():
    """
    Devuelve (run_id, modelo, escalador_X, escalador_y) listos para usar.

    Si MLFLOW_RUN_ID esta definido como variable de entorno, se intenta
    cargar ESE run directamente (sin buscar alternativas).

    Si no, se buscan las corridas del experimento ordenadas por rmse_test
    ascendente (menor error = mejor modelo) y se intenta CARGAR DE VERDAD
    cada una en ese orden (modelo + escaladores), quedandose con la primera
    que cargue sin errores.

    Se eligio intentar la carga real en vez de solo inspeccionar los nombres
    de las carpetas de artefactos (list_artifacts), porque MLflow 3.x puede
    registrar el modelo como una entidad de "Logged Model" que no siempre
    aparece como una carpeta tradicional en list_artifacts, aunque runs:/
    igual permita cargarlo correctamente. Probar la carga real es mas
    confiable que adivinar la estructura interna de artefactos.
    """
    cliente = mlflow.tracking.MlflowClient()

    if MLFLOW_RUN_ID:                                            # Si el usuario fijo un run puntual
        logger.info(f"Usando MLFLOW_RUN_ID fijado por variable de entorno: {MLFLOW_RUN_ID}")
        modelo, escalador_X, escalador_y = cargar_artefactos(MLFLOW_RUN_ID)
        return MLFLOW_RUN_ID, modelo, escalador_X, escalador_y

    experimento = cliente.get_experiment_by_name(MLFLOW_EXPERIMENT)  # Se busca el experimento por nombre

    if experimento is None:
        raise RuntimeError(
            f"No se encontro el experimento '{MLFLOW_EXPERIMENT}' en {MLFLOW_TRACKING_URI}. "
            "Verificar que el servidor MLflow este corriendo y que ya se haya "
            "ejecutado al menos un entrenamiento."
        )

    # Se traen las corridas ordenadas por rmse_test ascendente. Se pide un
    # numero generoso de resultados (no solo 1) para poder descartar las
    # que esten incompletas y seguir probando con las siguientes.
    corridas = cliente.search_runs(
        experiment_ids=[experimento.experiment_id],
        order_by=["metrics.rmse_test ASC"],
        max_results=200,
    )

    if not corridas:
        raise RuntimeError(
            f"El experimento '{MLFLOW_EXPERIMENT}' no tiene corridas registradas todavia."
        )

    # Se recorren las corridas en orden (de mejor a peor rmse_test) y se
    # intenta cargar cada una hasta que una funcione de verdad.
    for run in corridas:
        rmse_test = run.data.metrics.get("rmse_test")
        if rmse_test is None:
            continue                                              # Corrida sin metricas: se descarta

        try:
            modelo, escalador_X, escalador_y = cargar_artefactos(run.info.run_id)
            logger.info(
                f"Mejor run COMPLETO seleccionado automaticamente: {run.info.run_id} "
                f"(rmse_test={rmse_test})"
            )
            return run.info.run_id, modelo, escalador_X, escalador_y
        except Exception as error:
            logger.warning(
                f"Run {run.info.run_id} (rmse_test={rmse_test}) descartado: "
                f"no se pudo cargar completamente ({error}). Probablemente "
                "quedo incompleto por una ejecucion interrumpida."
            )

    # Si se llego hasta aca, ninguna corrida del experimento pudo cargarse
    raise RuntimeError(
        f"Ninguna corrida del experimento '{MLFLOW_EXPERIMENT}' pudo cargarse "
        "completamente (modelo + escaladores). Volve a ejecutar "
        "entrenamiento2.py sin interrumpirlo, o fija MLFLOW_RUN_ID con el "
        "run_id de una corrida que sepas que esta completa."
    )


# Se cargan modelo y escaladores UNA sola vez, al levantar la API (no en cada
# request), para que las predicciones sean rapidas. Si algo falla aca, la API
# no debe arrancar en un estado inconsistente, asi que se deja propagar el error.
RUN_ID_ACTIVO, MODELO, ESCALADOR_X, ESCALADOR_Y = seleccionar_y_cargar_modelo()

logger.info(f"Modelo y escaladores cargados correctamente desde el run {RUN_ID_ACTIVO}.")
logger.info(f"Features esperadas por el modelo ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

# ---------------------------------------------------------------------------
# 4. DEFINICION DINAMICA DEL ESQUEMA DE ENTRADA (validacion de datos)
# ---------------------------------------------------------------------------
# Se construye un modelo de Pydantic con un campo float por cada feature en
# FEATURE_COLUMNS. Esto evita hardcodear los nombres de columnas en este
# archivo: si transformations.py cambia, este esquema se actualiza solo.
#
# Pydantic v2, por defecto, YA RECHAZA:
#   - campos faltantes                         -> HTTP 422
#   - tipos invalidos (ej. un string "abc")     -> HTTP 422
#   - campos extra no declarados (con Config)   -> HTTP 422
# Lo que agregamos nosotros a mano es el rechazo explicito de NaN e infinito,
# porque un valor como float('nan') o float('inf') pasa la validacion de tipo
# "es un float" pero rompe silenciosamente la prediccion del modelo si no se
# controla (el resultado seria NaN sin ningun error claro para quien llama).

campos_dinamicos = {
    nombre_feature: (float, ...)                # (tipo, "..." = campo obligatorio)
    for nombre_feature in FEATURE_COLUMNS
}


def _validar_no_nan_ni_infinito(valor: float, nombre_campo: str) -> float:
    """Funcion compartida que rechaza NaN e infinito para cualquier campo."""
    if math.isnan(valor):
        raise ValueError(f"El campo '{nombre_campo}' no puede ser NaN.")
    if math.isinf(valor):
        raise ValueError(f"El campo '{nombre_campo}' no puede ser infinito.")
    return valor


# Se arma un validador de Pydantic por cada feature, reutilizando la misma
# funcion de chequeo de NaN/infinito. Se guardan en un diccionario y se pasan
# a create_model() via el parametro __validators__: agregarlos con setattr()
# DESPUES de crear el modelo no funciona en Pydantic v2 (el validador queda
# como atributo suelto, pero el motor de validacion interno del modelo nunca
# lo registra), por eso deben inyectarse en el momento de la creacion.
validadores_dinamicos = {}
for nombre_feature in FEATURE_COLUMNS:
    def _crear_validador(nombre_campo):
        # Closure para capturar correctamente el nombre de cada campo en el loop
        def _validador(cls, valor):
            return _validar_no_nan_ni_infinito(valor, nombre_campo)
        return _validador

    validadores_dinamicos[f"_validar_{nombre_feature}"] = field_validator(nombre_feature)(
        classmethod(_crear_validador(nombre_feature))
    )

EntradaPrediccion: type[BaseModel] = create_model(
    "EntradaPrediccion",                        # Nombre del modelo generado dinamicamente
    __validators__=validadores_dinamicos,          # Validadores de NaN/infinito por campo
    **campos_dinamicos,
)


class SalidaPrediccion(BaseModel):
    """Esquema de la respuesta exitosa del endpoint /predict."""
    target_next_hour_predicho: float             # La prediccion en unidades reales de C6H6_GT
    run_id_modelo: str                            # Que corrida de MLflow genero esta prediccion
    features_utilizadas: list[str]                 # Que columnas se usaron, para trazabilidad


# ---------------------------------------------------------------------------
# 5. APLICACION FASTAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de Prediccion C6H6_GT (proxima hora)",
    description=(
        "Sirve el mejor modelo de Red Neuronal (MLPRegressor) registrado en "
        "MLflow para predecir la concentracion de benceno una hora hacia el "
        "futuro, con validacion estricta de los datos de entrada."
    ),
    version="1.0.0",
)


# -- 5.1 Manejador de errores de validacion (campos invalidos, faltantes, etc.) --
@app.exception_handler(RequestValidationError)
async def manejador_errores_validacion(request: Request, exc: RequestValidationError):
    """
    Intercepta cualquier error de validacion (tipo incorrecto, campo
    faltante, NaN/infinito rechazado por nuestros validadores, etc.) y
    devuelve una respuesta 422 clara, listando cada campo problematico y
    el motivo, en vez del detalle crudo por defecto de FastAPI.
    """
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
        content={
            "error": "Datos de entrada invalidos.",
            "detalle": errores_legibles,
        },
    )


# -- 5.2 Manejador de errores inesperados (no filtra detalles internos) --------
@app.exception_handler(Exception)
async def manejador_errores_generico(request: Request, exc: Exception):
    """
    Ultima red de seguridad: si algo falla de forma inesperada durante la
    prediccion (por ejemplo, un problema al escalar los datos), se loguea el
    detalle completo del lado del servidor mediante logger.exception(...),
    pero al cliente se le devuelve un mensaje generico y controlado en vez
    de un traceback crudo (evita filtrar detalles internos de implementacion).
    """
    logger.exception("Error inesperado al procesar la solicitud.")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Ocurrio un error inesperado al procesar la solicitud."},
    )


# -- 5.3 Endpoint de salud, para verificar que la API y el modelo estan OK -----
@app.get("/health")
def salud():
    """Chequeo simple de que la API esta viva y el modelo quedo cargado."""
    return {
        "estado": "ok",
        "run_id_modelo": RUN_ID_ACTIVO,
        "experimento": MLFLOW_EXPERIMENT,
        "cantidad_features": len(FEATURE_COLUMNS),
    }


# -- 5.4 Endpoint principal de prediccion ---------------------------------------
@app.post("/predict", response_model=SalidaPrediccion)
def predecir(entrada: EntradaPrediccion):
    """
    Recibe los valores de las features (una fila) y devuelve la prediccion
    de target_next_hour (concentracion de C6H6_GT una hora hacia el futuro).

    Gracias al esquema dinamico EntradaPrediccion, FastAPI ya valido antes
    de llegar aca que: no falten campos, que todos sean numericos, y que
    ninguno sea NaN o infinito. Aca solo queda armar el DataFrame, escalar,
    predecir, y des-escalar el resultado.
    """
    # Se convierte el modelo de Pydantic a diccionario y luego a DataFrame de
    # una sola fila, respetando el MISMO orden de columnas con el que se
    # entreno el modelo (fundamental: un StandardScaler es sensible al orden).
    datos_dict = entrada.model_dump()
    fila = pd.DataFrame([[datos_dict[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    # Se escala la fila de entrada con el MISMO escalador usado en entrenamiento
    fila_escalada = ESCALADOR_X.transform(fila)

    # Prediccion en la escala interna del modelo (estandarizada)
    prediccion_escalada = MODELO.predict(fila_escalada)

    # Se revierte el escalado para obtener el valor en unidades reales de C6H6_GT
    prediccion_real = ESCALADOR_Y.inverse_transform(
        prediccion_escalada.reshape(-1, 1)
    ).ravel()[0]

    # Chequeo defensivo adicional: si por algun motivo el modelo devolviera
    # NaN o infinito (por ejemplo, datos de entrada validos pero muy fuera
    # del rango de entrenamiento), se informa como error claro al cliente
    # en vez de devolver un numero sin sentido con status 200.
    if math.isnan(prediccion_real) or math.isinf(prediccion_real):
        logger.error(f"El modelo devolvio un valor invalido: {prediccion_real}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "El modelo genero una prediccion invalida (NaN/infinito)."},
        )

    return SalidaPrediccion(
        target_next_hour_predicho=float(prediccion_real),
        run_id_modelo=RUN_ID_ACTIVO,
        features_utilizadas=FEATURE_COLUMNS,
    )
