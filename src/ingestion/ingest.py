#=====================================================================================
# Módulo de Ingesta de Datos desde SQL Server.
# Este script extrae la información de la tabla dbo.AirQuality, valida que contenga 
# información y la almacena localmente en un archivo CSV dentro de la carpeta data/raw.
#=====================================================================================

from pathlib import Path
import pandas as pd

# Importación del módulo interno para gestionar la conexión a la base de datos
from sql_server import get_engine

# Consulta SQL estándar para extraer todos los registros de la tabla de calidad del aire
QUERY = "SELECT * FROM dbo.AirQuality"

def ingest_data():
    
    # Define la ruta del directorio de destino (enfoque moderno y multiplataforma con pathlib)
    output_dir = Path("data/raw")
    
    # Crea la carpeta si no existe; 'parents=True' crea directorios intermedios si faltan
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define el nombre y la ruta final del archivo CSV que se va a generar
    output_file = output_dir / "air_quality_raw.csv"

    # Inicializa la variable de conexión en None para un control seguro en el bloque 'finally'
    engine = None

    try:
        print("Iniciando ingesta desde SQL Server...")

        # Establece la conexión utilizando la función personalizada importada
        engine = get_engine()


        # Ejecuta la consulta SQL y vuelca el resultado directamente en un DataFrame de Pandas
        df = pd.read_sql(QUERY, engine)

        # Validación: Si la tabla existe pero está vacía, frena el proceso lanzando un error controlado
        if df.empty:
            raise ValueError("La tabla existe pero esta vacia")

        # Exporta el DataFrame a un archivo CSV plano
        # 'index=False' evita que se agregue una columna extra con los índices numéricos de Pandas
        df.to_csv(
            output_file,
            index=False,
            encoding="utf-8" )# Asegura la correcta codificación de caracteres especiales

        # Mensajes de éxito informativos en la consola para monitoreo del script
        print("Ingesta completada correctamente.")
        print(f"Registros extraídos: {len(df)}")
        print(f"Columnas extraídas: {len(df.columns)}")
        print(f"Archivo RAW generado: {output_file}")

    except Exception as error:
        # Bloque de captura de errores: Atrapa cualquier falla en la conexión, consulta o escritura
        print("ERROR DE INGESTA")
        print(f"Detalle: {error}")
        raise # Eleva el error para que pueda ser detectado por herramientas de orquestación (ej. Airflow)

    finally:
        # Garantiza el cierre de la conexión a la base de datos sin importar si el script falló o tuvo éxito
        if engine is not None:
            engine.dispose()

# Punto de entrada estándar de Python: Asegura que el script solo se ejecute si se llama directamente
if __name__ == "__main__":
    ingest_data()
