#=====================================================================================
# Módulo de Ingesta de Datos desde SQL Server.
# Este script extrae la información de la tabla fuente dbo.AirQuality, valida que
# contenga información y realiza la ingesta de los datos sin transformar en la capa
# RAW/BRONZE, utilizando el schema bronze de SQL Server.
#=====================================================================================

import pandas as pd
from sqlalchemy import NVARCHAR

# Importación del módulo interno para gestionar la conexión a SQL Server
from sql_server import get_engine

# Tabla fuente desde donde se extraen los datos
SOURCE_QUERY = "SELECT * FROM dbo.AirQuality"

# Configuración de la capa RAW / BRONZE
BRONZE_SCHEMA = "bronze"
BRONZE_TABLE = "AirQuality"

# Columnas oficiales del dataset Air Quality
EXPECTED_COLUMNS = [
    "Date",
    "Time",
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
def ingest_data():
    engine = None
    try:
        print("Iniciando ingesta desde SQL Server...")

        # Establece la conexión con SQL Server
        engine = get_engine()

        # Extrae los datos desde la tabla fuente
        df = pd.read_sql(
            SOURCE_QUERY,
            engine
        )
        # Verifica que la fuente contenga registros
        if df.empty:
            raise ValueError(
                "La tabla dbo.AirQuality existe pero no contiene registros."
            )
        # Verifica que existan las columnas esperadas antes de cargar Bronze
        missing_columns = [
            column for column in EXPECTED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Faltan columnas en la fuente: {missing_columns}"
            )
        # Conserva únicamente las columnas oficiales
        df = df[EXPECTED_COLUMNS]

        # Mantiene las columnas Bronze como texto para conservar
        # los valores originales antes de Data Cleaning
        bronze_types = {
            column: NVARCHAR(length=50)
            for column in EXPECTED_COLUMNS
        }
        # Realiza la ingesta hacia la capa RAW / BRONZE
        # Los datos se almacenan sin limpieza ni transformaciones
        df.to_sql(
            name=BRONZE_TABLE,
            con=engine,
            schema=BRONZE_SCHEMA,
            if_exists="replace",
            index=False,
            dtype=bronze_types,
        )

        print("Ingesta completada correctamente.")
        print(f"Registros ingeridos: {len(df)}")
        print(f"Columnas ingeridas: {len(df.columns)}")
        print(
            f"Destino RAW / BRONZE: "
            f"{BRONZE_SCHEMA}.{BRONZE_TABLE}"
        )

    except Exception as error:
        print("ERROR DE INGESTA")
        print(f"Detalle: {error}")
        raise

    finally:
        if engine is not None:
            engine.dispose()

# Punto de entrada del proceso de ingesta
if __name__ == "__main__":
    ingest_data()
