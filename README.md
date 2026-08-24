# Grupo 8 — Air Quality MLOps

Proyecto de Machine Learning desarrollado bajo principios MLOps para el análisis y pronóstico de calidad del aire mediante técnicas de series de tiempo.
El objetivo final del proyecto es seleccionar una variable ambiental relevante y desarrollar un sistema de forecasting capaz de pronosticar su comportamiento.

## Integrantes

- José Vasquez Orozco
- Károl Godínez Solís
- Arlen Almansa Rodríguez

# Flujo actual del pipeline

Las etapas implementadas actualmente siguen el siguiente flujo:
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

Clonar el repositorio desde GitHub:

powershell git clone https://github.com/aparodriguez25-sketch/Grupo8_Air_Quality_MLOps.git

Ingresar al proyecto:powershell cd Grupo8_Air_Quality_MLOps

# 2. Crear el entorno virtual

Desde la raíz del proyecto ejecutar:
powershell python -m venv .venv

# 3. Activar el entorno virtual

En PowerShell: .\.venv\Scripts\Activate.ps1
Si el entorno se activó correctamente, la terminal deberá mostrar algo similar a:
(.venv) PS C:\Grupo8_Air_Quality_MLOps>

# 4. Instalar las dependencias

Con el entorno virtual activo ejecutar:
powershell: python -m pip install -r requirements.txt

Esto instalará las dependencias necesarias para ejecutar las etapas implementadas del pipeline.

# 5. Configurar SQL Server

El proyecto utiliza Microsoft SQL Server como fuente y almacenamiento de la capa RAW / Bronze.
La base de datos utilizada es: AirQuality
La fuente de ingesta es: dbo.AirQuality
La capa RAW / Bronze generada por el proceso es: bronze.AirQuality

El flujo es:
dbo.AirQuality
      ↓
ingest.py
      ↓
bronze.AirQuality

# 6. Configurar las variables de entorno

La configuración de conexión con SQL Server se administra mediante variables de entorno.
El repositorio contiene: .env.example
Crear una copia llamada:.env
En PowerShell puede realizarse mediante:
powershell Copy-Item .env.example .env

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
Desde la raíz del proyecto ejecutar:
powershell python src/ingestion/ingest.py

El script:

1. establece conexión con SQL Server;
2. consulta dbo.AirQuality;
3. obtiene los registros de la fuente;
4. verifica la estructura esperada;
5. carga los datos en la capa RAW / Bronze;
6. genera `bronze.AirQuality`;
7. informa la cantidad de registros y columnas procesadas.
Con los datos utilizados durante el desarrollo se obtuvo:

Iniciando ingesta desde SQL Server...
Ingesta completada correctamente.
Registros ingeridos: 9471
Columnas ingeridas: 15
Destino RAW / BRONZE: bronze.AirQuality

# 9. Verificar la capa RAW / Bronze

Después de ejecutar la ingesta se puede comprobar la cantidad de registros directamente desde SQL Server.
Ejecutar:sql
SELECT COUNT(*) AS total_registros
FROM bronze.AirQuality;
Resultado obtenido durante las pruebas: 9471

Esto permite comprobar que la capa RAW / Bronze fue generada correctamente.

# 10. Ejecutar Data Validation

Después de generar bronze.AirQuality, ejecutar desde la raíz del proyecto:
powershell python -m src.validation.validate

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

La ejecución actual: python -m src.validation.validate
produce un estado: WARNING

La alerta generada es:La validación detectó advertencias que deben revisarse.
Este resultado es esperado debido a los problemas de calidad identificados en el dataset RAW.

# 14. Verificación de los Quality Gates

Durante el desarrollo se verificaron los tres estados del sistema.

### PASS

Una prueba controlada produjo:
{'status': 'PASS', 'failed_rules': [], 'warning_rules': []}

### WARNING

La ejecución sobre los datos actuales produce:
status: WARNING
failed_rules: []
warning_rules:
- empty_rows
- encoded_missing_values

### FAIL

Una prueba controlada produjo:

{'status': 'FAIL', 'failed_rules': ['dataset_not_empty'], 'warning_rules': []}

También se comprobó que un FAIL detenga realmente la ejecución:
RuntimeError: La validación de datos falló. El pipeline debe detenerse.

Por lo tanto, el comportamiento verificado es:
PASS    → continúa
WARNING → alerta y continúa
FAIL    → alerta y detiene el pipeline

# 15. Ejecutar nuevamente el pipeline

Para reproducir las etapas implementadas actualmente, con el entorno configurado, ejecutar en este orden:

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

El diagnóstico completo de calidad, los resultados encontrados y las justificaciones de las decisiones tomadas se encuentran en: docs/data_quality_report.md

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

# 17. Estado actual del proyecto

Etapas completadas:
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

Etapas posteriores:

- [ ] Data Cleaning
- [ ] EDA
- [ ] Selección de variable ambiental objetivo
- [ ] Feature Engineering
- [ ] Entrenamiento del modelo de forecasting
- [ ] MLflow Tracking
- [ ] Evaluación
- [ ] Model Registry
- [ ] API
- [ ] Docker
- [ ] Monitoreo