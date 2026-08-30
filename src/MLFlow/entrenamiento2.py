"""
==============================================================================
ENTRENAMIENTO DE RED NEURONAL (MLP) CON TRACKING EN MLFLOW
==============================================================================
Objetivo:
    Entrenar el modelo de Red Neuronal (MLPRegressor) para predecir
    'target_next_hour' (C6H6_GT una hora hacia el futuro) y registrar todo
    el experimento en MLflow: parametros, metricas, graficos y el modelo
    entrenado (junto con su escalador), listo para ser versionado, comparado
    con otras corridas y desplegado desde el MLflow Model Registry.

Que queda registrado en MLflow por cada corrida (run):
    - Parametros del modelo (arquitectura, activacion, solver, etc.)
    - Metricas de evaluacion (RMSE, MAE, R2)
    - Graficos (real vs prediccion, dispersion) como artefactos
    - El modelo entrenado (formato sklearn nativo de MLflow, con firma e
      input_example) y el StandardScaler del target (necesario para
      des-escalar las predicciones en produccion)

Como ver los resultados:
    Ejecutar en la terminal, dentro de la carpeta del proyecto:
        mlflow ui, o accede a la interfaz del servidor MLflow directamente
        en http://127.0.0.1:5000 (donde ya debe estar corriendo el server)
    y abrir en el navegador la URL que indica (por defecto http://127.0.0.1:5000)
==============================================================================
"""
# ---------------------------------------------------------------------------
# 1. IMPORTACION DE LIBRERIAS
# ---------------------------------------------------------------------------
import pandas as pd                                   # Manejo de datos tabulares
import numpy as np                                     # Operaciones numericas
import matplotlib.pyplot as plt                          # Generacion de graficos
import mlflow                                            # Libreria principal de tracking de experimentos
import mlflow.sklearn                                    # Modulo especifico de MLflow para modelos scikit-learn
from mlflow.models.signature import infer_signature        # Infiere la "firma" (esquema de entrada/salida) del modelo

from sklearn.model_selection import train_test_split      # Division train/test
from sklearn.preprocessing import StandardScaler          # Escalado de variables
from sklearn.neural_network import MLPRegressor            # Modelo: Red Neuronal (Perceptron Multicapa)
from sklearn.metrics import (                                # Metricas de evaluacion de regresion
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import joblib                                              # Para serializar el escalador del target como artefacto

# Se reutiliza la unica implementacion oficial de preparacion de datos del
# proyecto, en vez de duplicar aqui la carga y limpieza del Excel. Esto
# garantiza que entrenamiento2.py use exactamente el mismo dataset (mismas
# features, mismo target, mismas filas excluidas) que el resto del pipeline.
import sys
import os

# entrenamiento2.py esta en Air_Quality_MLOps_MLFlow, un nivel debajo de la
# raiz del proyecto (donde esta 'src'). Se sube un nivel con dirname() para
# encontrarla, sin importar desde donde se ejecute el script.
RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RAIZ_PROYECTO)
from src.features.prepare_model_data import prepare_model_data
# Tambien se importan FEATURE_COLUMNS y MODEL_TARGET_COLUMN, que se usan
# mas abajo en la seccion 3 para definir las columnas de features y el
# target.
from src.features.transformations import FEATURE_COLUMNS, MODEL_TARGET_COLUMN

# El import de 'src' debe ir DESPUES de las lineas de sys.path.append,
# nunca antes, porque Python resuelve los imports en el momento en que los
# lee, y hasta ese punto 'src' todavia no es visible.

# Semilla fija para reproducibilidad de resultados
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 2. CONFIGURACION DE MLFLOW
# ---------------------------------------------------------------------------
# Se apunta a un servidor MLflow corriendo localmente (mlflow server) en
# http://127.0.0.1:5000, en vez de usar el backend de archivo/SQLite directo.
# Con esto, todas las corridas se registran contra ese servidor, y hay que
# tenerlo levantado ANTES de correr este script, por ejemplo con:
#     mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
# Si el servidor no esta corriendo, esta linea no falla al momento (la
# conexion es "perezosa"), pero el primer mlflow.log_param/log_metric mas
# abajo va a tirar un error de conexion (requests.exceptions.ConnectionError).
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Se agrupan todas las corridas de este problema bajo un mismo "experimento"
# para poder compararlas facilmente en la interfaz de MLflow.
mlflow.set_experiment("prediccion_C6H6_GT_red_neuronal")

# ---------------------------------------------------------------------------
# 3. CARGA Y PREPARACION DE DATOS (via src/features)
# ---------------------------------------------------------------------------
# prepare_model_data() ya se encarga de: ejecutar el feature engineering
# (build_features), quedarse solo con timestamp + FEATURE_COLUMNS + target,
# eliminar filas con NaN en features/target, y validar el orden cronologico.
# Por eso aqui ya NO se lee el Excel ni se ordena ni se filtra a mano.
df = prepare_model_data()

# Variable objetivo: concentracion de C6H6_GT una hora hacia el futuro.
# Se toma el nombre desde transformations.py para no duplicarlo como texto
# suelto ("target_next_hour") y evitar que ambos se desalineen en el futuro.
COLUMNA_TARGET = MODEL_TARGET_COLUMN

# Lista final de columnas usadas como variables predictoras (features),
# tambien centralizada en transformations.py junto con el target.
columnas_features = FEATURE_COLUMNS

# Matriz de features (X) y vector objetivo (y)
X = df[columnas_features].copy()
y = df[COLUMNA_TARGET].copy()

# Division train/test respetando el orden temporal (shuffle=False): se
# entrena con el pasado y se evalua con el futuro, como en un caso real.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% mas reciente para prueba
    shuffle=False,       # No mezclar: se preserva la secuencia temporal
)

# ---------------------------------------------------------------------------
# 4. ESCALADO DE VARIABLES (features y target)
# ---------------------------------------------------------------------------
# La Red Neuronal es sensible a la escala de los datos, por lo que se
# estandarizan tanto las features como el target (media 0, desviacion 1).
escalador_X = StandardScaler()                              # Escalador de las variables predictoras
X_train_esc = escalador_X.fit_transform(X_train)              # Se ajusta SOLO con datos de entrenamiento
X_test_esc = escalador_X.transform(X_test)                     # Se transforma el test con ese mismo ajuste

escalador_y = StandardScaler()                                # Escalador de la variable objetivo
y_train_esc = escalador_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_esc = escalador_y.transform(y_test.values.reshape(-1, 1)).ravel()

# ---------------------------------------------------------------------------
# 5. HIPERPARAMETROS DEL MODELO (varias configuraciones = varios runs)
# ---------------------------------------------------------------------------
# Se define una LISTA de
# configuraciones distintas. Cada elemento de la lista se entrena y se
# registra como una corrida (run) separada en MLflow, todas bajo el mismo
# experimento ("prediccion_C6H6_GT_red_neuronal"), lo que permite compararlas
# en la interfaz (tabla de runs) 
configuraciones_hiperparametros = [
    {
        "nombre_run": "MLP_1capa_32",
        "hidden_layer_sizes": (32,),       # Una sola capa oculta de 32 neuronas
        "activation": "relu",
        "solver": "adam",
        "max_iter": 2000,
        "early_stopping": True,
        "random_state": RANDOM_STATE,
    },
    {
        "nombre_run": "MLP_2capas_64_32",
        "hidden_layer_sizes": (64, 32),    # Configuracion original: 2 capas
        "activation": "relu",
        "solver": "adam",
        "max_iter": 2000,
        "early_stopping": True,
        "random_state": RANDOM_STATE,
    },
    {
        "nombre_run": "MLP_2capas_128_64",
        "hidden_layer_sizes": (128, 64),   # Red mas grande
        "activation": "relu",
        "solver": "adam",
        "max_iter": 2000,
        "early_stopping": True,
        "random_state": RANDOM_STATE,
    },
    {
        "nombre_run": "MLP_tanh_64_32",
        "hidden_layer_sizes": (64, 32),
        "activation": "tanh",              # Se cambia la funcion de activacion
        "solver": "adam",
        "max_iter": 2000,
        "early_stopping": True,
        "random_state": RANDOM_STATE,
    },
    {
        "nombre_run": "MLP_sgd_64_32",
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "sgd",                   # Se cambia el optimizador
        "max_iter": 2000,
        "early_stopping": True,
        "random_state": RANDOM_STATE,
    },
]

# ---------------------------------------------------------------------------
# 6. FUNCION AUXILIAR PARA CALCULAR METRICAS
# ---------------------------------------------------------------------------
def calcular_metricas(y_real, y_pred):
    """Calcula RMSE, MAE y R2 comparando valores reales contra predichos."""
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))    # Raiz del error cuadratico medio
    mae = mean_absolute_error(y_real, y_pred)                # Error absoluto medio
    r2 = r2_score(y_real, y_pred)                              # Coeficiente de determinacion
    return rmse, mae, r2

# ---------------------------------------------------------------------------
# 7. ENTRENAMIENTO: UNA CORRIDA (RUN) DE MLFLOW POR CADA CONFIGURACION
# ---------------------------------------------------------------------------
# Se recorre la lista de configuraciones definida en la seccion 5. Cada
# vuelta del "for" abre un "with mlflow.start_run(...)" propio, entrena un
# MLPRegressor con esos hiperparametros especificos, y registra todo
# (parametros, metricas, graficos, modelo) en una corrida independiente.
for config in configuraciones_hiperparametros:

    # Se separa el nombre del run del resto de los hiperparametros, ya que
    # "nombre_run" no es un hiperparametro valido de MLPRegressor.
    nombre_run = config["nombre_run"]
    hiperparametros = {k: v for k, v in config.items() if k != "nombre_run"}

    print(f"\n=== Entrenando configuracion: {nombre_run} ===")

    with mlflow.start_run(run_name=nombre_run) as run:

        # -- 7.1 Registrar los hiperparametros del modelo -----------------------
        mlflow.log_params(hiperparametros)                       # Se registran todos los hiperparametros de una vez

        # Tambien se registran metadatos utiles del experimento (no son
        # hiperparametros del modelo, pero ayudan a documentar la corrida)
       # mlflow.log_param("n_features", len(columnas_features))    # Cantidad de variables predictoras usadas
        #mlflow.log_param("features", ", ".join(columnas_features)) # Nombres de las features, como texto
        #mlflow.log_param("n_train", len(X_train))                    # Cantidad de registros de entrenamiento
        #mlflow.log_param("n_test", len(X_test))                       # Cantidad de registros de prueba
        #mlflow.log_param("test_size", 0.2)                              # Proporcion usada para el conjunto de prueba
        #mlflow.log_param("split_type", "cronologico_sin_mezcla")         # Se documenta que el split respeta el tiempo

        # -- 7.2 Entrenar el modelo ----------------------------------------------
        modelo_red = MLPRegressor(**hiperparametros)               # Se instancia el modelo con los hiperparametros de esta config
        modelo_red.fit(X_train_esc, y_train_esc)                     # Entrenamiento con datos escalados (features y target)

        # -- 7.3 Calcular metricas en train y test --------------------------------
        rmse_train, mae_train, r2_train = calcular_metricas(y_train, pred_train)  # Metricas sobre datos de entrenamiento
        rmse_test, mae_test, r2_test = calcular_metricas(y_test, pred_test)         # Metricas sobre datos de prueba

        # -- 7.4 Registrar las metricas en MLflow ----------------------------------
        # Metricas de entrenamiento (permiten detectar sobreajuste al compararlas con test)
        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("r2_train", r2_train)

        # Metricas de prueba (las mas importantes: reflejan el desempeno en datos nuevos)
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("mae_test", mae_test)
        mlflow.log_metric("r2_test", r2_test)
print("\nSe completaron todas las corridas del experimento.")
print("Para visualizar y comparar jjlos resultados, abrir en el navegador:")
print("    http://127.0.0.1:5000")