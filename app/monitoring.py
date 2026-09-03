"""
==============================================================================
MODULO DE MONITOREO (metricas Prometheus) PARA LA API DE PREDICCION
==============================================================================
Este modulo define las metricas que la API expone en el endpoint /metrics,
en formato que Prometheus sabe "scrapear" (leer periodicamente via HTTP).

Que se mide y por que:
    - Cuantos requests llegan, a que endpoint, y si terminaron bien o mal
      (permite armar alertas de "tasa de error alta" o "caida de trafico").
    - Cuanto tarda cada request (latencia) -> detecta degradacion de
      performance antes de que un usuario se queje.
    - La DISTRIBUCION de los valores predichos por el modelo
      (target_next_hour_predicho) -> si esa distribucion cambia mucho
      respecto a lo historico, puede ser una señal de "concept drift"
      (el mundo real cambio y el modelo ya no predice bien) o de un bug
      en el pipeline de features.
    - Si el modelo esta cargado o no (gauge 0/1) -> permite alertar
      inmediatamente si el contenedor arranco pero el modelo no cargo.
    - Errores por tipo (validacion, modelo no disponible, prediccion
      invalida) -> separar estos 3 tipos ayuda a diagnosticar mas rapido
      cual es la causa raiz cuando algo falla.

Cada metrica de Prometheus es uno de estos 3 tipos:
    - Counter: un numero que SOLO puede subir (ej. cantidad total de
      requests). Sirve para calcular tasas (requests/segundo) en Grafana.
    - Histogram: agrupa valores en "baldes" (buckets) para poder calcular
      percentiles despues (ej. "el 95% de los requests tardan menos de
      200ms", o "el 99% de las predicciones estan entre 2 y 15").
    - Gauge: un numero que puede subir Y bajar (ej. "esta el modelo
      cargado ahora mismo: 1 o 0").
==============================================================================
"""

import time                                                        # Para medir cuanto tarda cada request
from contextlib import contextmanager                                # Para armar un "cronometro" reutilizable

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
# Counter, Histogram, Gauge: los 3 tipos de metrica que se van a usar.
# generate_latest(): serializa todas las metricas registradas al formato
#   de texto que Prometheus espera leer en el endpoint /metrics.
# CONTENT_TYPE_LATEST: el "Content-Type" HTTP correcto para esa respuesta
#   (Prometheus lo exige para parsear bien la respuesta).

# ---------------------------------------------------------------------------
# 1. CONTADOR DE REQUESTS (por endpoint, metodo HTTP, y codigo de respuesta)
# ---------------------------------------------------------------------------
# Las "labels" (endpoint, metodo, codigo) permiten despues, en Prometheus/
# Grafana, filtrar y agrupar: por ejemplo "tasa de error 5xx en /predict
# en la ultima hora", sin tener que crear una metrica separada por caso.
REQUEST_COUNT = Counter(
    "api_requests_total",                    # Nombre de la metrica (convencion: sustantivo_total para Counters)
    "Cantidad total de requests HTTP recibidos",  # Descripcion (aparece en /metrics y en la UI de Prometheus)
    ["endpoint", "metodo", "codigo_http"],       # Labels: dimensiones por las que se puede filtrar despues
)

# ---------------------------------------------------------------------------
# 2. HISTOGRAMA DE LATENCIA (cuanto tarda cada request, en segundos)
# ---------------------------------------------------------------------------
REQUEST_LATENCY_SEGUNDOS = Histogram(
    "api_request_duration_seconds",           # Convencion: sustantivo_unidad para Histograms
    "Duracion de cada request HTTP, en segundos",
    ["endpoint"],                               # Se separa por endpoint (predict es mas lento que health)
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    # Los "buckets" son los limites usados para armar el histograma: cada
    # request cae en el primer balde cuyo limite supera su duracion. Estos
    # valores (10ms a 10s) son razonables para una API de inferencia liviana;
    # si el modelo fuera mas pesado, convendria correr el sistema primero y
    # ajustar los buckets a la distribucion real observada.
)

# ---------------------------------------------------------------------------
# 3. HISTOGRAMA DE VALORES PREDICHOS (para detectar drift en las salidas)
# ---------------------------------------------------------------------------
PREDICCION_VALOR = Histogram(
    "prediccion_target_next_hour",
    "Distribucion de los valores predichos de target_next_hour (concentracion de C6H6_GT)",
    buckets=(0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65),
    # Buckets elegidos en base al rango real observado en el dataset de
    # entrenamiento (min ~0.1, mediana ~8.1, max ~63.7, ver exploracion de
    # datos hecha al principio del proyecto). Si el modelo se reentrena con
    # datos de un rango muy distinto, conviene revisar estos limites.
)

# ---------------------------------------------------------------------------
# 4. ESTADO DEL MODELO (1 = cargado y listo, 0 = no disponible)
# ---------------------------------------------------------------------------
MODELO_DISPONIBLE = Gauge(
    "modelo_disponible",
    "1 si el modelo se cargo correctamente al iniciar la API, 0 si no",
)

# ---------------------------------------------------------------------------
# 5. CONTADOR DE ERRORES, DESGLOSADO POR TIPO
# ---------------------------------------------------------------------------
ERRORES_COUNT = Counter(
    "api_errores_total",
    "Cantidad de errores devueltos por la API, por tipo",
    ["tipo_error"],   # valores esperados: "validacion_422", "modelo_no_disponible_503", "prediccion_invalida_500"
)


@contextmanager
def medir_latencia(endpoint: str):
    """
    Cronometro reutilizable: se usa como
        with medir_latencia("/predict"):
            ... codigo del endpoint ...
    y automaticamente registra cuanto tardo ese bloque en el histograma
    REQUEST_LATENCY_SEGUNDOS, con el label del endpoint correspondiente.
    Evita repetir "start = time.time() / end = time.time()" en cada endpoint.
    """
    inicio = time.perf_counter()                # Marca de tiempo de alta precision, al entrar al bloque
    try:
        yield                                     # Acá se ejecuta el codigo del endpoint (el "with" en si)
    finally:
        duracion = time.perf_counter() - inicio    # Se calcula el tiempo transcurrido, pase lo que pase (incluso con excepcion)
        REQUEST_LATENCY_SEGUNDOS.labels(endpoint=endpoint).observe(duracion)  # Se registra en el histograma


def registrar_prediccion(valor_predicho: float) -> None:
    """Registra un valor predicho en el histograma, para poder ver su distribucion en Grafana."""
    PREDICCION_VALOR.observe(valor_predicho)


def registrar_error(tipo_error: str) -> None:
    """Incrementa el contador de errores para el tipo indicado."""
    ERRORES_COUNT.labels(tipo_error=tipo_error).inc()


def marcar_modelo_disponible(disponible: bool) -> None:
    """Actualiza el gauge de disponibilidad del modelo (1 o 0)."""
    MODELO_DISPONIBLE.set(1 if disponible else 0)


def registrar_request(endpoint: str, metodo: str, codigo_http: int) -> None:
    """Incrementa el contador de requests con sus labels correspondientes."""
    REQUEST_COUNT.labels(endpoint=endpoint, metodo=metodo, codigo_http=str(codigo_http)).inc()


def generar_metricas() -> tuple[bytes, str]:
    """
    Devuelve (cuerpo, content_type) listos para responder en el endpoint
    /metrics. Prometheus hace un GET a esa ruta periodicamente (por defecto
    cada 15s) y parsea este texto.
    """
    return generate_latest(), CONTENT_TYPE_LATEST
