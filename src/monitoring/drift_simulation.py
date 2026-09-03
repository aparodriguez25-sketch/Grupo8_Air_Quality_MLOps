import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.features.prepare_model_data import prepare_model_data


def cargar_datos():
	"""
	Carga el dataset preparado para modelado utilizando
	el mismo pipeline del proyecto.
	"""
	df = prepare_model_data()

	print("Dataset cargado correctamente")
	print(f"Filas: {df.shape[0]}")
	print(f"Columnas: {df.shape[1]}")
	print(f"Periodo: {df['timestamp'].min()} - {df['timestamp'].max()}")

	return df

def dividir_batches(df):
    """
    Divide los datos cronológicamente en un conjunto de referencia
    y tres batches que simulan datos recibidos en producción.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)

    fin_reference = int(n * 0.70)
    restante = n - fin_reference
    tamano_batch = restante // 3

    reference = df.iloc[:fin_reference].copy()
    batch_1 = df.iloc[fin_reference:fin_reference + tamano_batch].copy()
    batch_2 = df.iloc[
        fin_reference + tamano_batch:fin_reference + (2 * tamano_batch)
    ].copy()
    batch_3 = df.iloc[fin_reference + (2 * tamano_batch):].copy()

    print("\nDivisión temporal para simulación de producción")
    print(f"REFERENCE: {len(reference)} registros")
    print(f"BATCH 1: {len(batch_1)} registros")
    print(f"BATCH 2: {len(batch_2)} registros")
    print(f"BATCH 3: {len(batch_3)} registros")


    return reference, batch_1, batch_2, batch_3

def calcular_psi(reference, current, variable, bins=10):
    """
    Calcula el Population Stability Index (PSI) para comparar
    la distribución de una variable entre los datos de referencia
    y un batch de producción.
    """
    ref = reference[variable].dropna()
    cur = current[variable].dropna()

    limites = np.quantile(ref, np.linspace(0, 1, bins + 1))
    limites = np.unique(limites)
    limites[0] = -np.inf
    limites[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=limites)
    cur_counts, _ = np.histogram(cur, bins=limites)

    ref_pct = ref_counts / len(ref)
    cur_pct = cur_counts / len(cur)

    epsilon = 0.0001
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

    return psi

def clasificar_drift(psi):
    """
    Clasifica el nivel de drift según el valor del PSI.

    PSI < 0.10       -> OK
    0.10 <= PSI < 0.25 -> WARNING
    PSI >= 0.25      -> ALERT
    """
    if psi < 0.10:
        return "OK"
    elif psi < 0.25:
        return "WARNING"
    else:
        return "ALERT"


def simular_drift(batch, variable):
    """
    Simula un cambio controlado en la distribución de una variable.
    Se trabaja sobre una copia para no modificar los datos originales.
    """
    batch_simulado = batch.copy()

    desplazamiento = batch_simulado[variable].std() * 2

    batch_simulado[variable] = (
        batch_simulado[variable] + desplazamiento
    )

    return batch_simulado

    
def generar_grafica_psi(resultados):
    """
    Genera evidencia visual del PSI obtenido en cada batch.
    """
    carpeta_salida = Path("docs") / "monitoring"
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    nombres = list(resultados.keys())
    valores = list(resultados.values())

    plt.figure(figsize=(8, 5))
    plt.bar(nombres, valores)

    plt.axhline(0.10, linestyle="--", label="WARNING = 0.10")
    plt.axhline(0.25, linestyle="--", label="ALERT = 0.25")

    plt.title("Drift de C6H6_GT por batch")
    plt.ylabel("PSI")
    plt.xlabel("Batch de producción")
    plt.legend()
    plt.tight_layout()

    ruta = carpeta_salida / "psi_drift_c6h6.png"
    plt.savefig(ruta, dpi=150)
    plt.close()

    print(f"\nGráfica guardada en: {ruta}")

def generar_grafica_simulacion(psi_original, psi_simulado):
    """
    Compara visualmente el PSI del Batch 3 original
    con el Batch 3 después de provocar drift.
    """
    carpeta_salida = Path("docs") / "monitoring"
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    nombres = ["Batch 3 original", "Batch 3 simulado"]
    valores = [psi_original, psi_simulado]

    plt.figure(figsize=(8, 5))
    plt.bar(nombres, valores)

    plt.axhline(0.10, linestyle="--", label="WARNING = 0.10")
    plt.axhline(0.25, linestyle="--", label="ALERT = 0.25")

    plt.title("Simulación controlada de drift en C6H6_GT")
    plt.ylabel("PSI")
    plt.xlabel("Escenario")
    plt.legend()
    plt.tight_layout()

    ruta = carpeta_salida / "psi_drift_simulado.png"
    plt.savefig(ruta, dpi=150)
    plt.close()

    print(f"Gráfica de simulación guardada en: {ruta}")

if __name__ == "__main__":
    df = cargar_datos()
    reference, batch_1, batch_2, batch_3 = dividir_batches(df)

    variable = "C6H6_GT"

    psi_batch_1 = calcular_psi(reference, batch_1, variable)
    psi_batch_2 = calcular_psi(reference, batch_2, variable)
    psi_batch_3 = calcular_psi(reference, batch_3, variable)

    print("\nResultados PSI para C6H6_GT")
    print(f"BATCH 1: {psi_batch_1:.4f} - {clasificar_drift(psi_batch_1)}")
    print(f"BATCH 2: {psi_batch_2:.4f} - {clasificar_drift(psi_batch_2)}")
    print(f"BATCH 3: {psi_batch_3:.4f} - {clasificar_drift(psi_batch_3)}")

    resultados = {
        "Batch 1": psi_batch_1,
        "Batch 2": psi_batch_2,
        "Batch 3": psi_batch_3,
    }

    generar_grafica_psi(resultados)

    batch_3_simulado = simular_drift(batch_3, variable)

    psi_batch_3_simulado = calcular_psi(
        reference,
        batch_3_simulado,
        variable
    )

    print("\nSimulación controlada de drift")
    print(f"BATCH 3 original: {psi_batch_3:.4f} - {clasificar_drift(psi_batch_3)}")
    print(
        f"BATCH 3 simulado: {psi_batch_3_simulado:.4f} - "
        f"{clasificar_drift(psi_batch_3_simulado)}"
    )

    generar_grafica_simulacion(
        psi_batch_3,
        psi_batch_3_simulado
    )