"""
Pruebas sobre el MODELO ya exportado (model_artifact/), proyecto de
predicción de C6H6_GT (benceno, una hora hacia el futuro).

A diferencia del ejemplo de vinos (clasificación, 3 clases fijas), este es
un modelo de REGRESIÓN (MLPRegressor): no hay "clases válidas" ni
predict_proba. En su lugar se verifica que la predicción sea un número
real, finito, y que el pipeline completo (escalar entrada -> predecir ->
des-escalar salida) funcione de punta a punta, igual que lo hace
app/main.py en producción.

Tampoco se hardcodean valores de entrada por columna con nombres fijos
(como sí hacía el ejemplo de vinos, que conoce el orden y significado de
cada feature): las columnas se leen dinámicamente desde
model_artifact/feature_columns.json (el mismo archivo que usa app/main.py),
así el test queda sincronizado automáticamente si cambia el conjunto de
features.

Requiere que ya hayas corrido export_model.py (necesita la carpeta
model_artifact/, con modelo/, escaladores/ y feature_columns.json).

Correr con: pytest tests/test_model.py -v
"""

import json
import math
from pathlib import Path

import joblib
import pandas as pd
import mlflow.pyfunc
import pytest

MODEL_DIR = Path(__file__).resolve().parent.parent / "model_artifact"


def _carpeta_model_artifact_completa() -> bool:
    """Verifica que existan todas las piezas necesarias, no solo la carpeta raíz."""
    return (
        MODEL_DIR.exists()
        and (MODEL_DIR / "modelo").exists()
        and (MODEL_DIR / "escaladores" / "escalador_X.pkl").exists()
        and (MODEL_DIR / "escaladores" / "escalador_y.pkl").exists()
        and (MODEL_DIR / "feature_columns.json").exists()
    )


@pytest.fixture(scope="module")
def artefactos():
    """
    Carga modelo, escaladores y metadata de features UNA sola vez para
    todas las pruebas de este archivo. Si algo falta, se saltan las pruebas
    con un mensaje claro en vez de fallar con un traceback críptico.
    """
    if not _carpeta_model_artifact_completa():
        pytest.skip(
            "No existe model_artifact/ completo (modelo/, escaladores/, "
            "feature_columns.json). Corré primero: python export_model.py"
        )

    modelo = mlflow.pyfunc.load_model(str(MODEL_DIR / "modelo"))
    escalador_X = joblib.load(MODEL_DIR / "escaladores" / "escalador_X.pkl")
    escalador_y = joblib.load(MODEL_DIR / "escaladores" / "escalador_y.pkl")

    metadata = json.loads((MODEL_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    target_column = metadata["target_column"]

    return {
        "modelo": modelo,
        "escalador_X": escalador_X,
        "escalador_y": escalador_y,
        "feature_columns": feature_columns,
        "target_column": target_column,
    }


@pytest.fixture(scope="module")
def input_valido(artefactos):
    """
    Un input de ejemplo con todas las features en 0.5. No representa un
    valor físico realista (no conocemos los rangos válidos de cada feature,
    a diferencia del ejemplo de vinos), pero es suficiente para verificar
    que el pipeline de escalado/predicción/des-escalado funciona de punta
    a punta sin errores.
    """
    feature_columns = artefactos["feature_columns"]
    return pd.DataFrame([[0.5] * len(feature_columns)], columns=feature_columns)


def _predecir_valor_real(artefactos, input_df: pd.DataFrame) -> float:
    """Reproduce el mismo flujo que usa app/main.py: escalar -> predecir -> des-escalar."""
    fila_escalada = pd.DataFrame(
        artefactos["escalador_X"].transform(input_df),
        columns=artefactos["feature_columns"],
    )
    prediccion_escalada = artefactos["modelo"].predict(fila_escalada)
    valor_escalado = float(pd.Series(prediccion_escalada).iloc[0])
    return float(artefactos["escalador_y"].inverse_transform([[valor_escalado]])[0][0])


# ---------- CARGA ----------

def test_el_modelo_carga_sin_error(artefactos):
    assert artefactos["modelo"] is not None


def test_los_escaladores_cargan_sin_error(artefactos):
    assert artefactos["escalador_X"] is not None
    assert artefactos["escalador_y"] is not None


def test_hay_al_menos_una_feature_declarada(artefactos):
    assert len(artefactos["feature_columns"]) > 0


# ---------- CONSISTENCIA DE FORMA (evita errores silenciosos de escalado) ----------

def test_escalador_X_espera_la_misma_cantidad_de_features(artefactos):
    """
    Si feature_columns.json quedara desincronizado con el escalador (por
    ejemplo, si se reentrenó el modelo con otra cantidad de features y no
    se volvió a exportar), esta prueba lo detecta antes de que rompa en
    producción con un error de shape poco claro.
    """
    cantidad_features = len(artefactos["feature_columns"])
    assert artefactos["escalador_X"].n_features_in_ == cantidad_features, (
        f"El escalador_X espera {artefactos['escalador_X'].n_features_in_} "
        f"features, pero feature_columns.json declara {cantidad_features}"
    )


def test_escalador_y_espera_una_sola_columna(artefactos):
    assert artefactos["escalador_y"].n_features_in_ == 1


# ---------- PREDICCION ----------

def test_input_valido_produce_prediccion(artefactos, input_valido):
    fila_escalada = artefactos["escalador_X"].transform(input_valido)
    resultado = artefactos["modelo"].predict(fila_escalada)
    assert resultado is not None
    assert len(resultado) == 1


def test_prediccion_es_un_numero_finito(artefactos, input_valido):
    """
    A diferencia de clasificación (donde se valida pertenencia a un set de
    clases), acá se valida que la predicción sea un float real: ni NaN ni
    infinito, que es justamente lo que app/main.py rechaza con error 500
    si llegara a pasar.
    """
    prediccion_real = _predecir_valor_real(artefactos, input_valido)
    assert isinstance(prediccion_real, float)
    assert not math.isnan(prediccion_real), "La predicción es NaN"
    assert not math.isinf(prediccion_real), "La predicción es infinita"


def test_prediccion_es_determinista(artefactos, input_valido):
    """El mismo input debe dar siempre la misma predicción (sin aleatoriedad oculta)."""
    r1 = _predecir_valor_real(artefactos, input_valido)
    r2 = _predecir_valor_real(artefactos, input_valido)
    assert r1 == pytest.approx(r2), "La predicción no es determinista para el mismo input"


def test_prediccion_cambia_con_un_input_distinto(artefactos):
    """
    Sanity check básico: el modelo no debería devolver siempre el mismo
    valor sin importar la entrada (lo cual indicaría un modelo roto o mal
    cargado, por ejemplo con pesos sin entrenar).
    """
    feature_columns = artefactos["feature_columns"]
    input_a = pd.DataFrame([[0.1] * len(feature_columns)], columns=feature_columns)
    input_b = pd.DataFrame([[0.9] * len(feature_columns)], columns=feature_columns)

    pred_a = _predecir_valor_real(artefactos, input_a)
    pred_b = _predecir_valor_real(artefactos, input_b)

    assert pred_a != pytest.approx(pred_b), (
        "El modelo devuelve la misma predicción para inputs muy distintos; "
        "podría estar mal entrenado o mal cargado."
    )


# ---------- METADATA ----------

def test_run_id_de_origen_esta_registrado():
    """
    export_model.py guarda de qué corrida de MLflow salió el modelo
    exportado; util para trazabilidad (saber que versión esta sirviendo la API).
    """
    ruta_run_id = MODEL_DIR / "RUN_ID.txt"
    if not ruta_run_id.exists():
        pytest.skip("No existe RUN_ID.txt; corré primero: python export_model.py")
    contenido = ruta_run_id.read_text(encoding="utf-8").strip()
    assert len(contenido) > 0, "RUN_ID.txt existe pero está vacío"
