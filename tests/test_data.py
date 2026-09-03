"""
Pruebas sobre los DATOS de entrada del modelo (proyecto de calidad del aire,
predicción de C6H6_GT una hora hacia el futuro).

A diferencia del dataset de vinos (que viene embebido en scikit-learn), los
datos de este proyecto se obtienen a través del pipeline oficial:
    prepare_model_data() -> build_features() -> clean_data() -> SQL Server

Por lo tanto, para correr estas pruebas hace falta:
    - Tener pyodbc instalado
    - Tener un archivo .env en la raíz del proyecto con las credenciales de
      conexión (DB_SERVER, DB_DATABASE, DB_DRIVER)
    - Tener acceso de red al servidor SQL Server con la tabla bronze.AirQuality

No se hardcodean nombres de columnas ni rangos físicos "a mano": se importan
FEATURE_COLUMNS y MODEL_TARGET_COLUMN desde src.features.transformations (la
misma fuente de verdad que usa el resto del proyecto), para que estas pruebas
queden sincronizadas automáticamente si el equipo agrega o quita una feature.

Cubre: esquema, tipos, orden cronológico, missing values, infinitos,
variables obligatorias, no-negatividad de la variable objetivo (una
concentración de benceno no puede ser negativa).

Correr con: pytest tests/test_data.py -v
"""

import sys
import os

# Este archivo se asume ubicado en una carpeta 'tests/' en la raíz del
# proyecto (junto a 'src'). Si se ejecuta desde otra carpeta, ajustar el
# número de dirname() igual que se hizo en entrenamiento2.py / api_prediccion.py.
RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RAIZ_PROYECTO)

import numpy as np
import pandas as pd
import pytest

from src.features.prepare_model_data import prepare_model_data
from src.features.transformations import FEATURE_COLUMNS, MODEL_TARGET_COLUMN

# Columna de tiempo que siempre debe estar presente, además de las features
# y el target (prepare_model_data la conserva explícitamente).
COLUMNA_TIMESTAMP = "timestamp"

# Todas las columnas que el DataFrame de modelado debe tener, en cualquier
# orden (a diferencia del dataset de vinos, acá no exigimos un orden exacto
# de columnas porque prepare_model_data ya arma el DataFrame él mismo y el
# orden no es una garantía funcional del pipeline).
COLUMNAS_ESPERADAS = [COLUMNA_TIMESTAMP, *FEATURE_COLUMNS, MODEL_TARGET_COLUMN]


@pytest.fixture(scope="module")
def df():
    """
    Carga el DataFrame de modelado UNA sola vez para todas las pruebas de
    este archivo, usando la única implementación oficial del pipeline
    (prepare_model_data), igual que hace entrenamiento2.py.
    """
    return prepare_model_data()


# ---------- ESQUEMA ----------

def test_columnas_presentes(df):
    """El DataFrame debe contener, como mínimo, timestamp + features + target."""
    faltantes = set(COLUMNAS_ESPERADAS) - set(df.columns)
    assert not faltantes, f"Faltan columnas esperadas: {faltantes}"


def test_no_hay_dataset_vacio(df):
    """El pipeline no debería devolver un DataFrame sin filas."""
    assert len(df) > 0, "prepare_model_data() no devolvió ninguna fila"


# ---------- TIPOS ----------

def test_features_son_numericas(df):
    for col in FEATURE_COLUMNS:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} no es numérica"


def test_target_es_numerico(df):
    assert pd.api.types.is_numeric_dtype(df[MODEL_TARGET_COLUMN]), (
        f"{MODEL_TARGET_COLUMN} no es numérico"
    )


def test_timestamp_es_fecha(df):
    assert pd.api.types.is_datetime64_any_dtype(df[COLUMNA_TIMESTAMP]), (
        "La columna timestamp no tiene tipo de fecha/hora"
    )


# ---------- ORDEN CRONOLOGICO ----------

def test_timestamp_orden_cronologico(df):
    """
    Fundamental para un problema de series de tiempo: el dataset debe venir
    ordenado cronológicamente, o el split train/test (shuffle=False) y las
    features de lag/media móvil dejarían de tener sentido.
    """
    assert df[COLUMNA_TIMESTAMP].is_monotonic_increasing, (
        "El dataset no mantiene el orden cronológico"
    )


def test_no_hay_timestamps_duplicados(df):
    assert not df[COLUMNA_TIMESTAMP].duplicated().any(), (
        "Hay timestamps duplicados en el dataset"
    )


# ---------- MISSING / INFINITOS ----------

def test_no_hay_valores_faltantes_en_features(df):
    faltantes = df[FEATURE_COLUMNS].isnull().sum()
    columnas_con_nan = faltantes[faltantes > 0]
    assert columnas_con_nan.empty, f"Hay NaN en features:\n{columnas_con_nan}"


def test_no_hay_valores_faltantes_en_target(df):
    assert df[MODEL_TARGET_COLUMN].isnull().sum() == 0, (
        f"{MODEL_TARGET_COLUMN} no debería tener valores nulos"
    )


def test_no_hay_infinitos_en_features(df):
    assert np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=float)).all(), (
        "Hay valores infinitos en alguna de las features"
    )


def test_no_hay_infinitos_en_target(df):
    assert np.isfinite(df[MODEL_TARGET_COLUMN].to_numpy(dtype=float)).all(), (
        f"Hay valores infinitos en {MODEL_TARGET_COLUMN}"
    )


# ---------- RANGOS (solo lo que sabemos con certeza del dominio) ----------
# A diferencia del dataset de vinos, no conocemos de antemano el rango físico
# válido de cada feature individual (lags, medias móviles, hora/día/mes con
# nombres que pueden variar). Lo único que sabemos con certeza del dominio es
# que la variable objetivo representa una concentración de benceno, y una
# concentración no puede ser negativa.

def test_target_no_es_negativo(df):
    assert (df[MODEL_TARGET_COLUMN] >= 0).all(), (
        f"{MODEL_TARGET_COLUMN} tiene valores negativos, lo cual no es "
        "físicamente posible para una concentración de benceno"
    )


# ---------- VARIABLES OBLIGATORIAS ----------

@pytest.mark.parametrize("col", FEATURE_COLUMNS)
def test_feature_obligatoria_presente(df, col):
    """Cada feature declarada en FEATURE_COLUMNS es obligatoria: no puede faltar."""
    assert col in df.columns


def test_target_obligatorio_presente(df):
    assert MODEL_TARGET_COLUMN in df.columns


def test_cantidad_de_features_coincide_con_transformations(df):
    """
    Chequeo de consistencia: si alguien agrega una columna nueva al
    DataFrame pero se olvida de declararla en FEATURE_COLUMNS (o viceversa),
    esta prueba lo va a marcar como un cambio a revisar, no como un bug
    automático (a veces el DataFrame trae columnas auxiliares a propósito).
    """
    columnas_numericas_candidatas = set(df.select_dtypes(include=[np.number]).columns) - {MODEL_TARGET_COLUMN}
    columnas_no_declaradas = columnas_numericas_candidatas - set(FEATURE_COLUMNS)
    if columnas_no_declaradas:
        pytest.skip(
            f"Hay columnas numéricas en el DataFrame no declaradas en "
            f"FEATURE_COLUMNS: {columnas_no_declaradas}. Revisar si son "
            "features nuevas que faltan agregar a transformations.py."
        )
