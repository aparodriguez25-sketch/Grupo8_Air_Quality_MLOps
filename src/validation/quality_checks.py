import pandas as pd

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

# ==========================================================
# REGLA 1: DATASET NO VACÍO
# ==========================================================

def check_dataset_not_empty(df):
    # Comprueba que existan datos para realizar la validación.
    row_count = len(df)

    return {
        "passed": row_count > 0,
        "row_count": row_count,
    }

# ==========================================================
# REGLA 2: ESQUEMA DEL DATASET
# ==========================================================

def check_expected_columns(df):
    # Comprueba que el dataset contenga las columnas esperadas.
    actual_columns = list(df.columns)

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in actual_columns]

    extra_columns = [
        column for column in actual_columns
        if column not in EXPECTED_COLUMNS]

    return {
        "passed": not missing_columns and not extra_columns,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,}

# ==========================================================
# REGLA 3: FILAS DUPLICADAS
# ==========================================================

def check_duplicate_rows(df):
    # Excluye las filas completamente vacías antes de buscar duplicados.
    rows_with_data = df.dropna(how="all")
    duplicate_count = int(rows_with_data.duplicated().sum())

    return {
        "passed": duplicate_count == 0,
        "duplicate_count": duplicate_count,
    }

# ==========================================================
# REGLA 4: FILAS COMPLETAMENTE VACÍAS
# ==========================================================

def check_empty_rows(df):
    # Cuenta las filas donde todas las columnas están vacías.
    empty_rows = int(df.isna().all(axis=1).sum())

    return {
        "passed": empty_rows == 0,
        "empty_rows": empty_rows,
    }

# ==========================================================
# REGLA 5: FECHAS VÁLIDAS
# ==========================================================

def check_valid_dates(df):
    # Excluye filas completamente vacías antes de validar la fecha.
    rows_with_data = df.dropna(how="all")

    parsed_dates = pd.to_datetime(
        rows_with_data["Date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    invalid_dates = int(parsed_dates.isna().sum())

    return {
        "passed": invalid_dates == 0,
        "invalid_dates": invalid_dates,
    }

# ==========================================================
# REGLA 6: HORAS VÁLIDAS
# ==========================================================

def check_valid_times(df):
    # Excluye filas completamente vacías antes de validar la hora.
    rows_with_data = df.dropna(how="all")

    parsed_times = pd.to_datetime(
        rows_with_data["Time"],
        format="%H.%M.%S",
        errors="coerce"
    )

    invalid_times = int(parsed_times.isna().sum())

    return {
        "passed": invalid_times == 0,
        "invalid_times": invalid_times,
    }

# ==========================================================
# REGLA 7: VALORES NUMÉRICOS CONVERTIBLES
# ==========================================================

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

def check_numeric_columns(df):
    # Excluye filas completamente vacías antes de validar valores numéricos.
    rows_with_data = df.dropna(how="all")
    invalid_values = {}

    for column in NUMERIC_COLUMNS:
        normalized_values = (
            rows_with_data[column]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )

        converted_values = pd.to_numeric(
            normalized_values,
            errors="coerce"
        )

        invalid_count = int(converted_values.isna().sum())

        if invalid_count > 0:
            invalid_values[column] = invalid_count

    return {
        "passed": len(invalid_values) == 0,
        "invalid_values": invalid_values,
    }

# ==========================================================
# REGLA 8: MISSING VALUES CODIFICADOS COMO -200
# ==========================================================

def check_encoded_missing_values(df):
    # Excluye filas completamente vacías antes de buscar valores -200.
    rows_with_data = df.dropna(how="all")

    missing_by_column = {}

    for column in NUMERIC_COLUMNS:
        normalized_values = (
            rows_with_data[column]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )

        converted_values = pd.to_numeric(
            normalized_values,
            errors="coerce"
        )

        missing_count = int((converted_values == -200).sum())

        if missing_count > 0:
            missing_by_column[column] = missing_count

    return {
        "passed": True,
        "warning": len(missing_by_column) > 0,
        "encoded_missing_values": missing_by_column,
    }

# ==========================================================
# REGLA 9: CONTINUIDAD TEMPORAL
# ==========================================================

def check_temporal_continuity(df):
    # Excluye filas completamente vacías.
    rows_with_data = df.dropna(how="all").copy()

    timestamps = pd.to_datetime(
        rows_with_data["Date"].astype(str)
        + " "
        + rows_with_data["Time"].astype(str),
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce"
    )

    valid_timestamps = timestamps.dropna()

    expected_timestamps = pd.date_range(
        start=valid_timestamps.min(),
        end=valid_timestamps.max(),
        freq="h"
    )

    duplicate_timestamps = int(valid_timestamps.duplicated().sum())

    missing_timestamps = expected_timestamps.difference(
        pd.DatetimeIndex(valid_timestamps)
    )

    return {
        "passed": (
            duplicate_timestamps == 0
            and len(missing_timestamps) == 0
        ),
        "duplicate_timestamps": duplicate_timestamps,
        "missing_hours": len(missing_timestamps),
    }
