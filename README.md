# Grupo 8 — Air Quality MLOps

Proyecto de Machine Learning operacionalizado bajo principios MLOps para el pronóstico de calidad del aire.

## Integrantes
José Vasquez Orozco 
Károl Godínez Solís
Arlen Almansa Rodríguez

## Fuente de datos

El proyecto utiliza el dataset **Air Quality**, almacenado en Microsoft SQL Server para realizar la ingesta de datos de manera reproducible.

Configuración actual:

- Base de datos: `AirQuality`
- Tabla: `dbo.AirQuality`
- Motor: Microsoft SQL Server
- Autenticación: Windows Authentication

Los datos generados durante la ejecución del pipeline no se almacenan directamente en el repositorio Git.

## Configuración del entorno

Crear el entorno virtual:

```powershell
python -m venv .venv
```

Activar el entorno virtual en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias del proyecto:

```powershell
python -m pip install -r requirements.txt
```

## Configuración de SQL Server

La configuración de conexión se maneja mediante variables de entorno.

El archivo `.env.example` contiene la plantilla necesaria para configurar cada entorno:

```env
DB_SERVER=localhost
DB_DATABASE=AirQuality
DB_DRIVER=ODBC Driver 18 for SQL Server
DB_TRUSTED_CONNECTION=yes
DB_TRUST_SERVER_CERTIFICATE=yes
```

La configuración local de conexión se almacena en el archivo `.env`.

> El archivo `.env` está excluido del repositorio mediante `.gitignore` para evitar versionar configuraciones locales o información sensible.

## Ingesta de datos

La ingesta se realiza mediante un script reproducible que obtiene los datos directamente desde SQL Server.

Ejecutar desde la raíz del proyecto:

```powershell
python src/ingestion/ingest.py
```

El proceso:

1. Establece la conexión con SQL Server.
2. Consulta la tabla `dbo.AirQuality`.
3. Extrae los datos sin aplicar limpieza o transformaciones.
4. Genera el dataset RAW en `data/raw/air_quality_raw.csv`.
5. Detiene la ejecución y reporta el error si la ingesta falla.

En la prueba inicial se obtuvieron:

```text
Registros extraídos: 9471
Columnas extraídas: 15
```

Los archivos generados dentro de `data/raw/` están excluidos del repositorio Git para evitar almacenar directamente datasets de gran tamaño.

## Estructura actual del proyecto

```text
Grupo8_Air_Quality_MLOps/
│
├── data/
│   └── raw/
│       └── .gitkeep
│
├── src/
│   └── ingestion/
│       ├── __init__.py
│       ├── ingest.py
│       └── sql_server.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Estado actual

- [x] Repositorio Git configurado
- [x] Ramas `main`, `develop` y `feature/data-ingestion`
- [x] SQL Server configurado como fuente de datos
- [x] Entorno virtual y dependencias
- [x] Conexión Python con SQL Server
- [x] Ingesta reproducible
- [x] Generación de datos RAW
- [ ] Data Validation y Data Quality Gates
- [ ] Data Cleaning
- [ ] Feature Engineering
- [ ] Entrenamiento
- [ ] MLflow Tracking
- [ ] Evaluación y Model Registry
- [ ] API
- [ ] Docker
- [ ] Monitoreo