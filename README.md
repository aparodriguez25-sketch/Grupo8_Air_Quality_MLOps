# Grupo 8 — Air Quality MLOps

Proyecto de Machine Learning desarrollado bajo principios MLOps para el análisis y pronóstico de calidad del aire mediante técnicas de series de tiempo.

El objetivo final del proyecto es seleccionar una variable ambiental relevante y desarrollar un sistema de forecasting capaz de pronosticar su comportamiento.

## Integrantes

- José Vasquez Orozco
- Károl Godínez Solís
- Arlen Almansa Rodríguez

## Avance y estado de las etapas del proyecto

Etapas completadas se marcarán con **x**:

- [x] Repositorio Git
- [x] Configuración del entorno
- [x] Conexión reproducible con SQL Server
- [x] Ingesta reproducible
- [x] Capa RAW / Bronze
- [x] Data Validation
- [x] Diagnóstico de Data Quality
- [x] Data Quality Gates
- [x] Sistema de alertas PASS / WARNING / FAIL
- [x] Verificación de detención ante fallos críticos
- [x] Documentación del diagnóstico de calidad
- [x] Data Cleaning
- [x] EDA
- [x] Selección de variable ambiental objetivo
- [x] Feature Engineering
- [ ] Entrenamiento del modelo de forecasting
- [ ] MLflow Tracking
- [ ] Evaluación
- [ ] Model Registry
- [ ] API
- [ ] Docker
- [ ] Monitoreo

---

# Flujo actual del pipeline

Las etapas implementadas actualmente siguen el siguiente flujo:

dbo.AirQuality
      ↓
INGESTA
      ↓
bronze.AirQuality
      ↓
DATA VALIDATION
      ↓
DATA QUALITY CHECKS
      ↓
QUALITY GATES
      ↓
DATA CLEANING
      ↓
EDA
      ↓
FEATURE ENGINEERING
      ↓
DATASET PARA MODELADO

dbo.AirQuality funciona como fuente de datos.

bronze.AirQuality representa la capa RAW / Bronze utilizada por las etapas posteriores del pipeline.

# Requisitos previos

Para reproducir las etapas actuales del proyecto se requiere:

- Python
- Git
- Microsoft SQL Server
- ODBC Driver 18 for SQL Server
- Acceso a una instancia de SQL Server
- PowerShell o una terminal equivalente

El proyecto fue desarrollado utilizando un entorno virtual de Python.

# 1. Clonar el repositorio

Clonar el repositorio desde GitHub: powershell

git clone https://github.com/aparodriguez25-sketch/Grupo8_Air_Quality_MLOps.git

Ingresar al proyecto: powershell

cd Grupo8_Air_Quality_MLOps

# 2. Crear el entorno virtual

Desde la raíz del proyecto ejecutar: powershell

python -m venv .venv

# 3. Activar el entorno virtual

En PowerShell: .\.venv\Scripts\Activate.ps1

Si el entorno se activó correctamente, la terminal deberá mostrar algo similar a:

(.venv) PS C:\Grupo8_Air_Quality_MLOps>

# 4. Instalar las dependencias

Con el entorno virtual activo ejecutar: powershell

python -m pip install -r requirements.txt

Esto instalará las dependencias necesarias para ejecutar las etapas implementadas del pipeline.

---

# 5. Configurar SQL Server

El proyecto utiliza Microsoft SQL Server como fuente y almacenamiento de la capa RAW / Bronze.

La base de datos utilizada es:

AirQuality

La fuente de ingesta es:

dbo.AirQuality

La capa RAW / Bronze generada por el proceso es:

bronze.AirQuality

El flujo es:

dbo.AirQuality
      ↓
ingest.py
      ↓
bronze.AirQuality

# 6. Configurar las variables de entorno

La configuración de conexión con SQL Server se administra mediante variables de entorno.

El repositorio contiene:

.env.example

Crear una copia llamada:

.env

En PowerShell puede realizarse mediante: powershell

Copy-Item .env.example .env

El archivo deberá contener una configuración equivalente a:

.env
DB_SERVER=localhost
DB_DATABASE=AirQuality
DB_DRIVER=ODBC Driver 18 for SQL Server
DB_TRUSTED_CONNECTION=yes
DB_TRUST_SERVER_CERTIFICATE=yes

Modificar los valores cuando la configuración local de SQL Server sea diferente.

El archivo .env contiene configuración local y está excluido del repositorio mediante .gitignore.

No se debe subir este archivo al repositorio.

# 7. Verificar la fuente de datos

Antes de ejecutar la ingesta debe existir la tabla:

dbo.AirQuality

dentro de la base de datos:

AirQuality

Esta tabla constituye la fuente original utilizada por el script de ingesta.

La fuente esperada contiene 15 columnas correspondientes a las mediciones temporales, contaminantes, sensores y variables ambientales del dataset Air Quality.

# 8. Ejecutar la ingesta

Desde la raíz del proyecto ejecutar: powershell

python src/ingestion/ingest.py

El script:

1. establece conexión con SQL Server;
2. consulta dbo.AirQuality;
3. obtiene los registros de la fuente;
4. verifica la estructura esperada;
5. carga los datos en la capa RAW / Bronze;
6. genera bronze.AirQuality;
7. informa la cantidad de registros y columnas procesadas.

Con los datos utilizados durante el desarrollo se obtuvo:

Iniciando ingesta desde SQL Server...
Ingesta completada correctamente.
Registros ingeridos: 9471
Columnas ingeridas: 15
Destino RAW / BRONZE: bronze.AirQuality

# 9. Verificar la capa RAW / Bronze

Después de ejecutar la ingesta se puede comprobar la cantidad de registros directamente desde SQL Server.

Ejecutar: sql
SELECT COUNT(*) AS total_registros
FROM bronze.AirQuality;

Resultado obtenido durante las pruebas:  9471

Esto permite comprobar que la capa RAW / Bronze fue generada correctamente.

# 10. Ejecutar Data Validation

Después de generar bronze.AirQuality, ejecutar desde la raíz del proyecto: powershell

python -m src.validation.validate

La validación consulta directamente: bronze.AirQuality

# 11. Validaciones automáticas implementadas

Actualmente el pipeline ejecuta nueve reglas automáticas:

1. Dataset no vacío.
2. Presencia de las columnas esperadas.
3. Registros duplicados.
4. Filas completamente vacías.
5. Fechas válidas.
6. Horas válidas.
7. Convertibilidad de variables numéricas.
8. Valores faltantes codificados.
9. Continuidad temporal.

Las reglas se encuentran implementadas principalmente en:

src/validation/quality_checks.py

Su evaluación se realiza mediante:

src/validation/quality_gates.py

y las alertas mediante:

src/validation/alerts.py

# 12. Interpretar los Data Quality Gates

Los resultados de las reglas se clasifican mediante tres estados:

## PASS

Indica que no se detectaron fallos ni advertencias bajo las reglas configuradas.

PASS
 ↓
el pipeline puede continuar

## WARNING

Indica que existen problemas de calidad que requieren revisión, pero que no se consideran fallos críticos suficientes para detener inmediatamente el pipeline.

WARNING
   ↓
se genera una advertencia
   ↓
el pipeline puede continuar

## FAIL

Indica que una regla crítica no fue superada.

FAIL
 ↓
se genera una alerta crítica
 ↓
el pipeline se detiene

Ante un FAIL, validate.py lanza un RuntimeError.

# 13. Resultado esperado con los datos actuales

La ejecución: powershell

python -m src.validation.validate

produce un estado:

WARNING

La alerta generada es: La validación detectó advertencias que deben revisarse.

Este resultado es esperado debido a los problemas de calidad identificados en el dataset RAW.

# 14. Verificación de los Quality Gates

Durante el desarrollo se verificaron los tres estados del sistema.

### PASS

Una prueba controlada produjo: python
{   "status": "PASS",
    "failed_rules": [],
    "warning_rules": [] }

### WARNING

La ejecución sobre los datos actuales produce:

status: WARNING

failed_rules: []

warning_rules:
- empty_rows
- encoded_missing_values

### FAIL

Una prueba controlada produjo: python
{   "status": "FAIL",
    "failed_rules": ["dataset_not_empty"],
    "warning_rules": [] }

También se comprobó que un FAIL detenga realmente la ejecución:

RuntimeError: La validación de datos falló.
El pipeline debe detenerse.

Por lo tanto, el comportamiento verificado es:

PASS    → continúa
WARNING → alerta y continúa
FAIL    → alerta y detiene el pipeline

# 15. Ejecutar nuevamente el pipeline

Para reproducir las primeras etapas implementadas, con el entorno configurado, ejecutar:

powershell
python src/ingestion/ingest.py
python -m src.validation.validate

El flujo reproducido será:

dbo.AirQuality
      ↓
INGESTA
      ↓
bronze.AirQuality
      ↓
DATA VALIDATION
      ↓
DATA QUALITY CHECKS
      ↓
QUALITY GATES
      ↓
PASS / WARNING / FAIL

# 16. Documentación de Data Quality

El diagnóstico completo de calidad, los resultados encontrados y las justificaciones de las decisiones tomadas se encuentran en:

docs/data_quality_report.md

Este documento incluye el análisis de:

- valores faltantes;
- valores -200;
- duplicados;
- registros vacíos;
- fechas y horas;
- continuidad temporal;
- tipos de datos;
- valores físicamente imposibles;
- outliers;
- cardinalidad;
- skewness;
- correlaciones;
- anomalías estadísticas;
- leakage;
- imbalance;
- errores de unidad;
- estructura temporal relevante para forecasting.

No se eliminaron o imputaron datos automáticamente durante esta etapa sin analizar primero su impacto.

# 17. Etapa 5 — Data Cleaning

La limpieza de datos se implementa mediante funciones reutilizables ubicadas en:

src/cleaning/
├── __init__.py
├── clean.py
└── transformations.py

La etapa utiliza como entrada los datos almacenados en:

bronze.AirQuality

y aplica el siguiente proceso:

bronze.AirQuality
        ↓
Conversión de variables numéricas
        ↓
Normalización de coma decimal
        ↓
Conversión de -200 a NaN
        ↓
Eliminación de filas completamente vacías
        ↓
Creación de timestamp
        ↓
Dataset preparado para EDA

## Ejecutar Data Cleaning

Desde la raíz del proyecto y con el entorno virtual activo: powershell

python -m src.cleaning.clean

Resultado esperado:

Data Cleaning completado correctamente.
Filas originales: 9471
Filas después de limpieza: 9357
Filas eliminadas: 114
Columnas: 16

La limpieza:

- convierte las variables ambientales y sensores a formato numérico;
- normaliza valores con coma decimal;
- convierte los valores -200 a NaN;
- elimina únicamente las 114 filas completamente vacías;
- conserva registros con valores faltantes parciales;
- crea la columna temporal timestamp;
- conserva Date y Time para trazabilidad.

En esta etapa no se realiza todavía la imputación definitiva de valores faltantes ni la eliminación automática de outliers.

Estas decisiones se analizan posteriormente mediante evidencia obtenida durante el EDA.

Las etapas posteriores pueden reutilizar la misma lógica mediante: python

from src.cleaning.clean import clean_data

df = clean_data()

De esta manera, EDA, Feature Engineering y entrenamiento utilizan la misma lógica de preparación de datos.

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
- no se utiliza backfill;
- solamente se rellenan hasta tres horas consecutivas;
- los bloques largos permanecen como NaN;
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

Con SQL Server y .env configurados, ejecutar:
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

Para comenzar esta etapa desde develop:

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

# En el notebook modeling se hicierón los siguietnes pasos

- Importación de librerías: Se immportan las librerías necesarias para hacer el modelado de datos.
- Carga de datos. Se preocede a cargar el dataframe mediante la función prepare_model_data que se encuentra en la siguiente ruta D:\Ciencia de Datos\06 Proyecto Integrador\Leccion 12\Proyecto final\Grupo8_Air_Quality_MLOps\src\features\prepare_model_data.py
- Definición de variables predictorias (X) y variable objetivo  (y): Se crean las variables x concentracion de C6H6_GT y los features que son predictoras y la variable objetivo que es target_next_hour
- División en conjunto de entrenamiento y prueba: X y y se dividen para X_train, X_test, y_train, y_test y en este caso se usa un 20% de los datos para test y el 80% restante para train. Obtenemos la siguiente distribución.

Registros de entrenamiento: 6579
Registros de prueba: 1645

- Escalado de variables: Se usa el método StandardScaler(), para escalar los modelos de red neuronal y SVM que son sensibles a la escala de datos. Arbol de decisión no es sensible a la escala de datos.
- Entrenamiento de los modelos: Se usa la función DecisionTreeRegressor para el módelo de árbol y tambien el método fit para entrenamiento. Se usa la función MLPRegressor para la red neuronal e igualmente el método fit para entrenar el módelo. Para el módelo SVM se uso la función SVR para el modelado y la función fit, para entrenar el modelo
- Cálculo de métricas (RMSE, MAE, R2) para cada modelo: Se crea la función calcular_metricas(y_real, y_pred) y posteriormente se llama a esta función para calcular las métricas con nuestros datos. Adicionalmente se concluye que el mejor modelo a utilizar es las redes neuronales.
- Gráfico del mejor modelo: Se usa la libretia de matplotlib para generar un gráfico donde se detalla el mejor modelo a utilizar y lo compara con las otras métricas

# 28. Integración con MLFlow

Como primerm paso debemos ejecutar el siguiente comando para crear un entorno virtual en Visual Studio Code

python -m venv Air_Quality_MLOps_MLFlow

Ahora tenemos que acticar el virtual enviroment, pero si nos da un error de seguridad primero debemos correr el siguiente 

comando Set-ExcecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Seguidamente activamos con el comando 

.\Air_Quality_MLOps_MLFlow\Scripts\Activate.ps1

Seguidamente en el entorno virtual corremos el siguiente comando

pip install mlflow scikit-learn pandas matplotlib 

Ahora vamos a correr el servidor de tracking de MLFlow y la baase de datos mlflow.db con el comando 

mlflow sercer --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifact
--host 127.0.0.1 --port 5000

Ahora en en el directorio \Air_Quality_MLOps_MLFlow creamos un archivo llamado entrenamiento2.py

En ese archivo de entrenamiento en forma resumida se crea el df usando prepare_model_data(). Se asignal los features y la variable target. Se genera un 20% de la información para pruebas. Se hace el escalado de las variables porque el modelo de red neuronal es sensible a la escala de datos. Se crean los hiperparametros del modelo. Se hace un escalado de variables debido a que las redes neuronales son sensibles a la escala de datos. Se crean los hiperparametros para generar mas de 1 run en MLFlow.  Se crea una función llamada calcular_metricas para rmse, mae, r2. Se crea el código para crear 5 runs en MLFlow. Se registran los parametros para MLFlow. Se entrena el modelo. Se calcula las metricasc en train y test. Por último se registran las métricas en MLFlow.

Ahora lo que debemos de hacer es correr el script entrenamiento2.py. Usamos el siguietne comando

python .\Air_Quality_MLOps_MLFlow\entrenamiento.py

Se debe obtener el siguiente output.

Se completaron todas las corridas del experimento.
Para visualizar y comparar jjlos resultados, abrir en el navegador:
    http://127.0.0.1:5000


# 29. API Fast

En el script api_prediccion.py se realizan los siguientes procesos. Se inportan las librerias y se configura para acceder a MLFlow. Se carga el modelo y los escaladores desde MLFlow. Se elige el modelo con menor error que es el mejor modelo. Se procede hacer manejo de excepciones para datos erróneos. Se procede a crear la APIFast tambión con manejo de errores.

Para correr API Fast se debe seguir los siguientes pasos


    1. Instalar dependencias (con el entorno virtual activado):
        pip install fastapi "uvicorn[standard]" mlflow joblib pandas numpy
    2. Tener el servidor MLflow levantado (con al menos una corrida
       registrada exitosamente), por ejemplo:
        mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
    3. Ejecutar la API (desde la raiz del proyecto, junto a la carpeta src):
        uvicorn api_prediccion:app --reload --port 8000
    4. Abrir la documentacion interactiva en:
        http://127.0.0.1:8000/docs





