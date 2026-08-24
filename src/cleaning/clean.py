import pandas as pd
from src.cleaning.transformations import (
    normalize_numeric_columns,
    replace_encoded_missing_values,
    remove_fully_empty_rows,
    create_timestamp,
)
from src.ingestion.sql_server import get_engine

# Consulta de la capa RAW / Bronze.
BRONZE_QUERY = "SELECT * FROM bronze.AirQuality"

def clean_data():
    engine = None
    try:
        # Establece la conexión con SQL Server.
        engine = get_engine()

        # Carga los datos desde la capa Bronze.
        df = pd.read_sql(
            BRONZE_QUERY,
            engine,
        )

        # Normaliza las columnas numéricas.
        cleaned_df = normalize_numeric_columns(df)

        # Convierte los valores -200 en NaN.
        cleaned_df = replace_encoded_missing_values(cleaned_df)

        # Elimina únicamente las filas completamente vacías.
        cleaned_df = remove_fully_empty_rows(cleaned_df)

        # Construye el timestamp para el análisis de series de tiempo.
        cleaned_df = create_timestamp(cleaned_df)

        # Muestra un resumen del resultado de limpieza.
        print("Data Cleaning completado correctamente.")
        print(f"Filas originales: {len(df)}")
        print(f"Filas después de limpieza: {len(cleaned_df)}")
        print(f"Filas eliminadas: {len(df) - len(cleaned_df)}")
        print(f"Columnas: {len(cleaned_df.columns)}")

        return cleaned_df

    finally:
        # Libera los recursos de conexión.
        if engine is not None:
            engine.dispose()

if __name__ == "__main__":
    clean_data()