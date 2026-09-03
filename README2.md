# Proyecto Integrador — Servicio de inferencia (MLflow + Docker + FastAPI)

Clasificador de vinos (`RandomForestClassifier`, scikit-learn) entrenado y versionado con MLflow,
servido como API REST dentro de un contenedor Docker autocontenido.

## Arquitectura

```
proyecto-mlops/
├── app/
│   └── main.py          # API FastAPI (carga el modelo UNA vez al iniciar)
├── model_artifact/       # Modelo exportado desde el MLflow Registry (generado por export_model.py)
├── export_model.py       # Script que exporta el modelo del Registry a model_artifact/
├── requirements.txt      # Dependencias con versión fija
├── Dockerfile
└── .dockerignore
```

**Decisión de diseño clave:** el contenedor NO se conecta a un `mlflow server` en tiempo de
inferencia. El modelo se exporta una sola vez a `model_artifact/` (carpeta autocontenida con
pesos + metadata) y esa carpeta se copia dentro de la imagen. Esto elimina la dependencia de
red/infraestructura externa durante `docker run` y cumple con "no depende de mi computadora sí
funciona": el mismo contenedor corre igual en cualquier máquina con Docker.

## 1. Exportar el modelo (una sola vez, en tu entorno local)

Con tu `mlflow server` corriendo y el modelo ya registrado:

```bash
pip install mlflow scikit-learn
python export_model.py
```

Esto crea la carpeta `model_artifact/` junto al `Dockerfile`. Ajusta `MODEL_URI` dentro de
`export_model.py` si tu modelo tiene otro nombre/alias.

## 2. Construir la imagen

```bash
docker build -t grupoX-mlops .
```

## 3. Levantar el servicio

```bash
docker run -p 8000:8000 grupoX-mlops
```

La API queda disponible en `http://localhost:8000`. Documentación interactiva automática en
`http://localhost:8000/docs` (Swagger UI, generada por FastAPI).

## 4. Probar el servicio

**Health check:**

```bash
curl http://localhost:8000/health
```

**Predicción:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2,
    "magnesium": 100, "total_phenols": 2.65, "flavanoids": 2.76,
    "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28, "color_intensity": 4.38,
    "hue": 1.05, "od280/od315_of_diluted_wines": 3.4, "proline": 1050
  }'
```

Respuesta esperada:

```json
{
  "prediction": 1,
  "probability": 0.87,
  "model_version": "3"
}
```

## Criterios de evaluación — cómo los cubre este proyecto

| Criterio | Cómo se cumple |
|---|---|
| **Reproducibilidad** | `requirements.txt` con versiones fijas; modelo empaquetado dentro de la imagen (no depende de servicios externos en runtime) |
| **Dependencias** | Únicamente las necesarias para servir el modelo (FastAPI, uvicorn, mlflow, scikit-learn, numpy, pydantic) — no se incluyen librerías de entrenamiento/notebook |
| **Tamaño razonable** | Imagen base `python:3.10-slim` (no la imagen completa); `.dockerignore` excluye `mlruns/`, `mlartifacts/`, notebooks y entornos virtuales |
| **Funcionamiento** | `docker build` + `docker run` levantan el servicio de punta a punta; `HEALTHCHECK` integrado; endpoint `/health` para verificación |
| **Documentación** | Este README explica arquitectura, pasos de build/run y ejemplos de request/response |

## Notas

- Cambia `MODEL_VERSION` en `app/main.py` cada vez que reexportes una nueva versión del modelo.
- Si tu equipo entrenó con una versión distinta de `mlflow` o `scikit-learn`, alinea esas versiones
  en `requirements.txt` con las que usaste al entrenar — un mismatch de versión de scikit-learn al
  deserializar el modelo (`pickle`) es la causa más común de que el contenedor falle al arrancar.

## Pruebas (tests)

```
tests/
├── test_data.py    # esquema, tipos, rangos, missing, variables obligatorias
├── test_model.py   # input válido -> prediction válida
└── test_api.py     # request -> 200 -> schema válido, + manejo de input inválido
```

### Instalar dependencias de desarrollo

```bash
pip install -r requirements-dev.txt
```

(`requirements-dev.txt` incluye `requirements.txt` + `pytest`, `httpx`, `pandas` — estas
últimas NO van dentro de la imagen Docker final, para no aumentar su tamaño con
herramientas que no se usan en producción.)

### Correr toda la suite

```bash
pytest tests/ -v
```

### Correr solo un archivo

```bash
pytest tests/test_data.py -v
pytest tests/test_model.py -v
pytest tests/test_api.py -v
```

Las pruebas de `test_model.py` y `test_api.py` se **saltan automáticamente** (no fallan)
si todavía no existe `model_artifact/` — así el equipo puede correr `test_data.py` desde
el primer día, antes incluso de tener un modelo entrenado.

### Qué pasa frente a un input inválido (evidencia para la rúbrica)

`test_api.py` prueba 6 escenarios de input inválido y confirma en cada uno que la API
responde `422 Unprocessable Entity` (no un 500, no un crash, no una predicción silenciosa
con datos basura) **antes** de que el request llegue al modelo:

| Escenario | Ejemplo | Resultado esperado |
|---|---|---|
| Falta variable obligatoria | sin `alcohol` | `422` + detalle de qué campo falta |
| Tipo de dato incorrecto | `alcohol: "trece"` | `422` |
| Valor fuera de rango (negativo) | `alcohol: -5.0` | `422` |
| Valor fuera de rango (extremo) | `magnesium: 99999` | `422` |
| Body vacío | `{}` | `422` |
| Mensaje de error informativo | — | el campo que falló aparece en `detail` |

Para verlo en vivo (con el contenedor corriendo):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"alcohol": -5.0, "malic_acid": 1.78, ...}'
```

Respuesta:

```json
{
  "detail": [
    {
      "type": "greater_than",
      "loc": ["body", "alcohol"],
      "msg": "Input should be greater than 0",
      "input": -5.0
    }
  ]
}
```
