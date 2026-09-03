# Imagen base ligera (no la "full" de Python, para mantener el tamaño razonable)
# Se usa 3.14 en vez de 3.10 porque requirements.txt fue generado con pip freeze
# desde un entorno local en Python 3.14, y fija versiones exactas (por ejemplo
# numpy==2.5.2) que requieren Python >=3.12 y no tienen wheels para 3.10.
FROM python:3.12-slim

WORKDIR /app

# Dependencias primero: aprovecha el cache de Docker.
# Si solo cambias el código, esta capa no se reconstruye.
COPY requirements1.txt .
RUN pip install --no-cache-dir -r requirements1.txt

# Código de la API y modelo ya exportado (autocontenido, sin dependencia de red)
COPY app/ ./app/
COPY model_artifact/ ./model_artifact/

EXPOSE 8000

# Healthcheck: docker/orquestadores pueden saber si el servicio está vivo
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
