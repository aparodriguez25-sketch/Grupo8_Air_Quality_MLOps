import numpy as np
import pandas as pd

# Columnas que deben convertirse a valores numéricos.
NUMERIC_COLUMNS = [
    "CO_GT",
    "PT08_S1_CO",
    "NMHC_GT",
    "C6H6_GT",
    "PT08_S2_NMHC",
    "NOx_GT",
    "PT08_S3_NOx",
    "NO2_GT",
    "PT08_S4_NO2",
    "PT08_S5_O3",
    "T",
    "RH",
    "AH",
]

# Convierte las columnas numéricas desde su representación RAW
# a valores numéricos utilizables por las etapas posteriores.
def normalize_numeric_columns(df):
    cleaned_df = df.copy()

    for column in NUMERIC_COLUMNS:
        # Convierte la coma decimal a punto y después convierte a número.
        cleaned_df[column] = pd.to_numeric(
            cleaned_df[column]
            .astype(str)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )

    return cleaned_df


# Convierte los valores -200 en NaN.
# En este dataset -200 representa ausencia de medición.
# Esta función no imputa datos, solamente normaliza los faltantes.
def replace_encoded_missing_values(df):
    cleaned_df = df.copy()

    cleaned_df[NUMERIC_COLUMNS] = cleaned_df[NUMERIC_COLUMNS].replace(
        -200,
        np.nan,
    )

    return cleaned_df

# Elimina únicamente las filas que no contienen ninguna información útil.
# No elimina registros que tengan valores faltantes parciales.
def remove_fully_empty_rows(df):
    cleaned_df = df.copy()

    cleaned_df = cleaned_df.dropna(
        how="all"
    )

    return cleaned_df

# Construye una columna timestamp a partir de Date y Time.
# Mantiene Date y Time originales para conservar trazabilidad.
def create_timestamp(df):
    cleaned_df = df.copy()

    cleaned_df["timestamp"] = pd.to_datetime(
        cleaned_df["Date"].astype(str)
        + " "
        + cleaned_df["Time"].astype(str).str.replace(".", ":", regex=False),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    return cleaned_df