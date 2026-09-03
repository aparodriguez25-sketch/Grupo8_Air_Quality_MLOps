"""
Pruebas sobre la API (FastAPI), sin necesidad de Docker: se prueba la app
directamente en memoria con TestClient.

Cubre: request válido -> HTTP 200 -> schema de respuesta válido,
       y qué pasa frente a distintos tipos de input inválido.

Diferencias clave respecto al ejemplo de vinos:
    - Es un modelo de REGRESIÓN: la respuesta trae
      "target_next_hour_predicho" (float), no "prediction"/"probability".
    - Los errores de validación acá vienen en el formato
      {"error": "...", "detalle": [{"campo": ..., "motivo": ...}]}
      (definido por el exception_handler propio de app/main.py), no en el
      formato por defecto de FastAPI {"detail": [...]}.
    - No se conocen rangos físicos válidos por feature (a diferencia del
      vino, donde "alcohol negativo" o "magnesio 99999" son inválidos por
      motivos de dominio conocidos). En su lugar, la API valida que ningún
      valor sea NaN o infinito, así que las pruebas de "rango inválido" se
      reemplazan por pruebas de NaN/infinito.
    - Las columnas de entrada se leen dinámicamente desde FEATURE_COLUMNS
      (expuesto por app/main.py), no se hardcodean nombres específicos.

Correr con: pytest tests/test_api.py -v
"""

import json as json_module

import pytest
from fastapi.testclient import TestClient

from app.main import app, FEATURE_COLUMNS

client = TestClient(app)

# Input válido de ejemplo: todas las features en 0.5. No representa un
# escenario físico realista (no conocemos rangos válidos por feature), pero
# alcanza para probar que el pipeline completo responde 200 sin errores.
INPUT_VALIDO = {nombre: 0.5 for nombre in FEATURE_COLUMNS}


@pytest.fixture(autouse=True)
def verificar_modelo_cargado(request):
    """
    Antes de cada test, verifica que el modelo esté cargado y lo salta si
    no lo está -- EXCEPTO para los dos tests diseñados específicamente para
    probar el escenario de modelo roto (más abajo), que necesitan correr
    justamente cuando el modelo NO carga. Por eso el fixture es de scope
    "function" (por test, no una sola vez por módulo): así puede decidir
    caso por caso en vez de saltear el archivo entero a la primera falla.
    """
    pruebas_que_esperan_modelo_roto = {
        "test_health_reporta_503_si_el_modelo_no_cargo",
        "test_predict_reporta_503_si_el_modelo_no_cargo",
    }
    if request.node.name in pruebas_que_esperan_modelo_roto:
        return

    resp = client.get("/health")
    if resp.status_code != 200:
        pytest.skip("El modelo no está cargado (¿corriste export_model.py?)")


# ---------- CASO FELIZ: request válido -> 200 -> schema válido ----------

def test_health_responde_200():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_con_input_valido_responde_200():
    resp = client.post("/predict", json=INPUT_VALIDO)
    assert resp.status_code == 200


def test_predict_respeta_el_schema_de_respuesta():
    resp = client.post("/predict", json=INPUT_VALIDO)
    body = resp.json()
    assert set(body.keys()) == {"target_next_hour_predicho", "model_version"}
    assert isinstance(body["target_next_hour_predicho"], float)
    assert isinstance(body["model_version"], str)


def test_prediccion_es_un_numero_finito():
    resp = client.post("/predict", json=INPUT_VALIDO)
    valor = resp.json()["target_next_hour_predicho"]
    assert valor == valor           # descarta NaN (NaN != NaN es True)
    assert valor not in (float("inf"), float("-inf"))


# ---------- INPUT INVÁLIDO: qué debe pasar en cada caso ----------

def test_falta_una_variable_obligatoria():
    """Quitar la primera feature del request: debe rechazarse, no debe intentar predecir."""
    primera_feature = FEATURE_COLUMNS[0]
    input_incompleto = INPUT_VALIDO.copy()
    del input_incompleto[primera_feature]
    resp = client.post("/predict", json=input_incompleto)
    assert resp.status_code == 422  # Unprocessable Entity: error de validación
    assert "detalle" in resp.json()


def test_tipo_de_dato_incorrecto():
    """Mandar texto donde se espera un número."""
    primera_feature = FEATURE_COLUMNS[0]
    input_invalido = INPUT_VALIDO.copy()
    input_invalido[primera_feature] = "no-es-un-numero"
    resp = client.post("/predict", json=input_invalido)
    assert resp.status_code == 422


def test_valor_nan_rechazado():
    """
    A diferencia del ejemplo de vinos (que rechaza por rango físico), este
    proyecto no conoce rangos válidos por feature, así que en su lugar
    valida que ningún valor sea NaN. Se envía como JSON crudo porque el
    estándar JSON no admite NaN literal al serializar con algunas librerías;
    Python sí puede emitirlo, y el servidor debe rechazarlo igual.
    """
    primera_feature = FEATURE_COLUMNS[0]
    input_invalido = INPUT_VALIDO.copy()
    input_invalido[primera_feature] = float("nan")
    cuerpo_crudo = json_module.dumps(input_invalido)
    resp = client.post("/predict", content=cuerpo_crudo, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


def test_valor_infinito_rechazado():
    primera_feature = FEATURE_COLUMNS[0]
    input_invalido = INPUT_VALIDO.copy()
    input_invalido[primera_feature] = float("inf")
    cuerpo_crudo = json_module.dumps(input_invalido)
    resp = client.post("/predict", content=cuerpo_crudo, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


def test_body_vacio():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_mensaje_de_error_es_informativo():
    """El error 422 debe indicar QUÉ campo falló, no solo que algo falló."""
    primera_feature = FEATURE_COLUMNS[0]
    input_invalido = INPUT_VALIDO.copy()
    input_invalido[primera_feature] = "no-es-un-numero"
    resp = client.post("/predict", json=input_invalido)
    detalle = resp.json()["detalle"]
    campos_reportados = [str(err.get("campo")) for err in detalle]
    assert any(primera_feature in campo for campo in campos_reportados)


# ---------- MODELO NO DISPONIBLE (503, en vez de 500 crudo) ----------
# Nota: estas dos pruebas no son parte del fixture autouse de arriba a
# propósito, ya que evaluan el comportamiento cuando el modelo SI está roto,
# a diferencia de las demás que asumen que está sano. Se saltan solas si el
# modelo esta cargado correctamente (nada que probar en ese caso).

def test_health_reporta_503_si_el_modelo_no_cargo():
    from app.main import load_error
    if load_error is None:
        pytest.skip("El modelo cargó correctamente; no aplica este escenario.")
    resp = client.get("/health")
    assert resp.status_code == 503


def test_predict_reporta_503_si_el_modelo_no_cargo():
    from app.main import load_error
    if load_error is None:
        pytest.skip("El modelo cargó correctamente; no aplica este escenario.")
    resp = client.post("/predict", json=INPUT_VALIDO)
    assert resp.status_code == 503
