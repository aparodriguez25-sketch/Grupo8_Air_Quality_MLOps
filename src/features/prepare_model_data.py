from src.features.build_features import build_features
from src.features.transformations import (
    FEATURE_COLUMNS,
    MODEL_TARGET_COLUMN,
)
# Prepara el dataset definitivo que será utilizado
# posteriormente para Train, Validation y Test.
def prepare_model_data():
    # Ejecuta la única implementación oficial
    # de Feature Engineering.
    df = build_features()

    # Columnas necesarias para modelado.
    model_columns = [
        "timestamp",
        *FEATURE_COLUMNS,
        MODEL_TARGET_COLUMN,
    ]
    model_df = df[model_columns].copy()
    rows_before = len(model_df)

    # Elimina únicamente filas que no pueden utilizarse
    # para entrenamiento porque falta alguna feature
    # requerida o el target real.
    model_df = (
        model_df
        .dropna(
            subset=FEATURE_COLUMNS + [MODEL_TARGET_COLUMN]
        )
        .reset_index(drop=True)
    )
    rows_after = len(model_df)

    # Valida que todas las features estén completas.
    if model_df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            "El dataset final contiene NaN en las features."
        )
    # El target nunca debe estar imputado ni vacío.
    if model_df[MODEL_TARGET_COLUMN].isna().any():
        raise ValueError(
            "El dataset final contiene NaN en el target."
        )
    # Mantiene el orden temporal.
    if not model_df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "El dataset final no mantiene el orden cronológico."
        )
    print("Dataset de modelado preparado correctamente.")
    print(f"Filas originales: {rows_before}")
    print(f"Filas finales: {rows_after}")
    print(f"Filas excluidas: {rows_before - rows_after}")
    print(f"Features predictivas: {len(FEATURE_COLUMNS)}")
    print(f"Target: {MODEL_TARGET_COLUMN}")
    print(
        f"Cobertura final: "
        f"{round(rows_after / rows_before * 100, 2)} %"
    )
    return model_df
if __name__ == "__main__":
    prepare_model_data()