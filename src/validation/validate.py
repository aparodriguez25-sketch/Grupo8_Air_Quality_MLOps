import pandas as pd
from src.ingestion.sql_server import get_engine
from src.validation.quality_checks import (
    check_dataset_not_empty,
    check_expected_columns,
    check_duplicate_rows,
    check_empty_rows,
    check_valid_dates,
    check_valid_times,
    check_numeric_columns,
    check_encoded_missing_values,
    check_temporal_continuity,
)
from src.validation.alerts import generate_validation_alert
from src.validation.quality_gates import evaluate_quality_gates

# ==========================================================
# CONFIGURACIÓN DE DATA VALIDATION
# ==========================================================

BRONZE_QUERY = "SELECT * FROM bronze.AirQuality"


# ==========================================================
# EJECUCIÓN DE DATA VALIDATION
# ==========================================================

def validate_data():
    # Inicializa el engine para poder cerrarlo de forma segura.
    engine = None

    try:
        # Establece la conexión con SQL Server.
        engine = get_engine()

        # Carga directamente los datos RAW desde la capa Bronze.
        df = pd.read_sql(
            BRONZE_QUERY,
            engine,
        )
        # Ejecuta todas las reglas de calidad.
        results = {
            "dataset_not_empty": check_dataset_not_empty(df),
            "expected_columns": check_expected_columns(df),
            "duplicate_rows": check_duplicate_rows(df),
            "empty_rows": check_empty_rows(df),
            "valid_dates": check_valid_dates(df),
            "valid_times": check_valid_times(df),
            "numeric_columns": check_numeric_columns(df),
            "encoded_missing_values": check_encoded_missing_values(df),
            "temporal_continuity": check_temporal_continuity(df),
        }
        # Evalúa las reglas y determina PASS, WARNING o FAIL.
        quality_status = evaluate_quality_gates(results)

        # Genera la alerta correspondiente al estado obtenido.
        alert = generate_validation_alert(quality_status)

        # Detiene el pipeline si existe un fallo crítico de calidad.
        if quality_status["status"] == "FAIL":
            raise RuntimeError(alert["message"])

        # PASS y WARNING pueden continuar.
        return {
            "results": results,
            "quality_status": quality_status,
            "alert": alert,
        }
    finally:
        # Libera los recursos de conexión con SQL Server.
        if engine is not None:
            engine.dispose()

# Punto de entrada de Data Validation.
if __name__ == "__main__":
    validation_result = validate_data()
    print(validation_result)