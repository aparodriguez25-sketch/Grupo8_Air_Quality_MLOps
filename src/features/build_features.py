from src.cleaning.clean import clean_data
from src.features.transformations import (
    validate_temporal_data,
    impute_short_gaps,
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_future_target,
)
# Construye el dataset de features reutilizando una sola lógica
# para experimentación, entrenamiento y producción.
def build_features():
    # Obtiene los datos preparados por Data Cleaning.
    df = clean_data()

    # Valida y ordena la estructura temporal.
    features_df = validate_temporal_data(df)

    # Crea el target futuro ANTES de imputar C6H6_GT.
    # Así target_next_hour conserva únicamente valores reales observados.
    features_df = create_future_target(
        features_df,
        horizon=1,
    )
    # Imputa únicamente huecos cortos del predictor C6H6_GT
    # utilizando información pasada.
    features_df = impute_short_gaps(
        features_df,
        limit=3,
    )
    # Crea variables temporales derivadas de timestamp.
    features_df = create_time_features(features_df)

    # Crea variables lag de la variable objetivo.
    features_df = create_lag_features(features_df)

    # Crea estadísticas móviles utilizando únicamente información pasada.
    features_df = create_rolling_features(features_df)

    # Muestra un resumen del resultado.
    print("Feature Engineering completado correctamente.")
    print(f"Filas: {len(features_df)}")
    print(f"Columnas: {len(features_df.columns)}")

    return features_df

if __name__ == "__main__":
    build_features()