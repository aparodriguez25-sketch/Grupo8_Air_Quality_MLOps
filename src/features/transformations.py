import pandas as pd

# Variable ambiental seleccionada durante el EDA.
TARGET_COLUMN = "C6H6_GT"

# Features seleccionadas para el primer modelo.
FEATURE_COLUMNS = [
    "C6H6_GT",
    "hour",
    "day_of_week",
    "month",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_168",
    "rolling_mean_3",
    "rolling_mean_24",
    "rolling_std_24",
]
# Target utilizado durante el entrenamiento.
MODEL_TARGET_COLUMN = "target_next_hour"

# Valida y ordena la estructura temporal antes de crear features.
def validate_temporal_data(df):
    if "timestamp" not in df.columns:
        raise ValueError(
            "El dataset no contiene la columna 'timestamp'."
        )
    features_df = df.copy()

    # Convierte timestamp a datetime.
    features_df["timestamp"] = pd.to_datetime(
        features_df["timestamp"],
        errors="raise",
    )
    # Ordena cronológicamente.
    features_df = (
        features_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    # Valida timestamps duplicados.
    if features_df["timestamp"].duplicated().any():
        raise ValueError(
            "Existen timestamps duplicados."
        )
    # Valida frecuencia horaria continua.
    differences = (
        features_df["timestamp"]
        .diff()
        .dropna()
    )

    if not (
        differences == pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "La serie temporal contiene intervalos "
            "diferentes de una hora."
        )
    return features_df

# Crea el objetivo de forecasting una hora hacia el futuro.
# El target se crea antes de cualquier imputación para conservar
# únicamente mediciones reales observadas.
def create_future_target(df, horizon=1):
    features_df = df.copy()
    features_df[MODEL_TARGET_COLUMN] = (
        features_df[TARGET_COLUMN]
        .shift(-horizon)
    )
    return features_df

# Imputa únicamente huecos cortos del predictor C6H6_GT.
# Utiliza forward fill y por lo tanto solo emplea información pasada.
def impute_short_gaps(df, limit=3):
    features_df = df.copy()
    features_df[TARGET_COLUMN] = (
        features_df[TARGET_COLUMN]
        .ffill(limit=limit)
    )
    return features_df

# Crea variables temporales a partir de timestamp.
def create_time_features(df):
    features_df = df.copy()
    features_df["hour"] = (
        features_df["timestamp"].dt.hour
    )
    features_df["day_of_week"] = (
        features_df["timestamp"].dt.dayofweek
    )
    features_df["month"] = (
        features_df["timestamp"].dt.month
    )
    return features_df

# Crea variables lag utilizando únicamente información histórica.
def create_lag_features(df):
    features_df = df.copy()
    features_df["lag_1"] = (
        features_df[TARGET_COLUMN]
        .shift(1)
    )
    features_df["lag_2"] = (
        features_df[TARGET_COLUMN]
        .shift(2)
    )
    features_df["lag_3"] = (
        features_df[TARGET_COLUMN]
        .shift(3)
    )
    features_df["lag_24"] = (
        features_df[TARGET_COLUMN]
        .shift(24)
    )
    features_df["lag_168"] = (
        features_df[TARGET_COLUMN]
        .shift(168)
    )
    return features_df

# Crea estadísticas móviles utilizando únicamente información pasada.
# shift(1) evita utilizar el valor actual de C6H6_GT.
def create_rolling_features(df):
    features_df = df.copy()

    historical_target = (
        features_df[TARGET_COLUMN]
        .shift(1)
    )
    # Promedio de las 3 horas anteriores.
    features_df["rolling_mean_3"] = (
        historical_target
        .rolling(
            window=3,
            min_periods=3,
        )
        .mean()
    )
    # Promedio de las 24 horas anteriores.
    features_df["rolling_mean_24"] = (
        historical_target
        .rolling(
            window=24,
            min_periods=24,
        )
        .mean()
    )
    # Desviación estándar de las 24 horas anteriores.
    features_df["rolling_std_24"] = (
        historical_target
        .rolling(
            window=24,
            min_periods=24,).std()
    )
    return features_df