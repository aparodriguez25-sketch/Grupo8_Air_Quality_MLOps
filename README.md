
---

# 18. Etapa 6 — Exploratory Data Analysis (EDA)

El análisis exploratorio se realizó reutilizando el dataset generado por Data Cleaning.

El notebook utilizado se encuentra en:

notebooks/01_eda_air_quality.ipynb

Durante el EDA se analizaron:

- estructura temporal;
- valores faltantes;
- estadísticas descriptivas;
- distribuciones;
- contaminantes;
- sensores;
- variables ambientales;
- correlaciones;
- comportamiento temporal;
- autocorrelación;
- patrones horarios, diarios y mensuales.

A partir del análisis se seleccionó como variable ambiental objetivo: C6H6_GT

correspondiente a la concentración de benceno.

Se identificó una dependencia temporal importante, incluyendo aproximadamente:

Autocorrelación lag 1 hora:  0.839
Autocorrelación lag 24 horas: 0.632

Estos resultados respaldan la utilización de información histórica para el problema de forecasting.

# 19. Objetivo de forecasting

Se definió como objetivo:

Predecir la concentración de C6H6_GT una hora hacia el futuro utilizando únicamente información disponible hasta la hora actual.

La variable objetivo utilizada para modelado es: target_next_hour

El problema queda definido como:

Información disponible en t
           ↓
         MODELO
           ↓
C6H6_GT en t + 1 hora

# 20. Etapa 7 — Feature Engineering

La implementación reutilizable de Feature Engineering se encuentra en:

src/features/
├── __init__.py
├── transformations.py
├── build_features.py
└── prepare_model_data.py

La lógica está centralizada para evitar diferencias entre las transformaciones utilizadas durante experimentación, entrenamiento y producción.

Para ejecutar Feature Engineering: python -m src.features.build_features

Resultado obtenido:

Data Cleaning completado correctamente.
Filas originales: 9471
Filas después de limpieza: 9357
Filas eliminadas: 114
Columnas: 16

Feature Engineering completado correctamente.
Filas: 9357
Columnas: 28

# 21. Features creadas

Se generan variables temporales: hour, day_of_week, month

Variables de rezago:
lag_1
lag_2
lag_3
lag_24
lag_168

Estadísticas móviles: rolling_mean_3, rolling_mean_24, rolling_std_24

Y el target de forecasting: target_next_hour

# 22. Tratamiento de valores faltantes

Después de Data Cleaning, C6H6_GT presentaba 366 valores faltantes.

Para huecos cortos se implementó una imputación causal mediante:
python ffill(limit=3)

La imputación utiliza únicamente información pasada.

target_next_hour se crea antes de realizar esta imputación y no se imputa.

Por lo tanto:
- no se utiliza información futura;
- no se utiliza `backfill`;
- solamente se rellenan hasta tres horas consecutivas;
- los bloques largos permanecen como `NaN`;
- el target conserva únicamente observaciones reales.

Después de la imputación:

C6H6_GT NaN antes:   366
C6H6_GT NaN después: 324

# 23. Features seleccionadas para el primer modelo

Se seleccionaron inicialmente 12 features predictivas:
C6H6_GT, hour, day_of_week, month, lag_1, lag_2
lag_3, lag_24, lag_168, rolling_mean_3,rolling_mean_24
rolling_std_24

La variable objetivo es: target_next_hour
timestamp se conserva para mantener el orden cronológico y la trazabilidad, pero no forma parte de las 12 features predictivas.

# 24. Dataset preparado para modelado

La preparación final del dataset se realiza mediante:

python -m src.features.prepare_model_data

El flujo implementado es:

clean_data()
      ↓
build_features()
      ↓
prepare_model_data()
      ↓
Dataset para Train / Validation / Test

Resultado obtenido:

Dataset de modelado preparado correctamente.
Filas originales: 9357
Filas finales: 8224
Filas excluidas: 1133
Features predictivas: 12
Target: target_next_hour
Cobertura final: 87.89 %

Por lo tanto, se entregan a la etapa de modelado:

8,224 observaciones
12 features predictivas
1 target

# 25. Documentación de Feature Engineering

Las decisiones y justificaciones técnicas se encuentran documentadas en:
docs/feature_engineering.md

Incluye:

- validación temporal;
- variables temporales;
- lags;
- estadísticas móviles;
- prevención de Data Leakage;
- definición del target;
- imputación causal;
- selección de features;
- preparación del dataset para modelado.

# 26. Reproducir el pipeline actual

Con SQL Server y `.env` configurados, ejecutar:
powershell
python src/ingestion/ingest.py
python -m src.validation.validate
python -m src.cleaning.clean
python -m src.features.build_features
python -m src.features.prepare_model_data

El pipeline disponible actualmente es:
SQL SERVER
     ↓
INGESTA
     ↓
RAW / BRONZE
     ↓
DATA VALIDATION
     ↓
DATA CLEANING
     ↓
EDA
     ↓
FEATURE ENGINEERING
     ↓
DATASET PARA MODELADO
     ↓
MODELADO Y ENTRENAMIENTO
(siguiente etapa)

# 27. Continuación del proyecto — Modelado y entrenamiento

La siguiente etapa debe utilizar directamente la preparación reutilizable existente:python
from src.features.prepare_model_data import prepare_model_data
df = prepare_model_data()

El DataFrame contiene:

- timestamp;
- 12 features predictivas;
- target_next_hour.

La división Train / Validation / Test deberá respetar el orden cronológico para evitar Data Leakage.

Los siguientes pasos previstos son:

1. división cronológica Train / Validation / Test;
2. construcción de un modelo baseline;
3. entrenamiento de modelos candidatos;
4. evaluación de métricas;
5. comparación de modelos;
6. selección del mejor modelo;
7. integración con MLflow;
8. registro de experimentos y artefactos.

Para comenzar esta etapa desde `develop`:

powershell
git switch develop
git pull origin develop
git switch -c feature/model-training

# Estado actual

Data Ingestion        ✅
Data Validation       ✅
Data Quality          ✅
Data Cleaning         ✅
EDA                   ✅
Feature Engineering   ✅
Dataset para modelado ✅

Model Training        ⏳
El punto de entrada recomendado para continuar es:
python:
from src.features.prepare_model_data import prepare_model_data
df = prepare_model_data()
