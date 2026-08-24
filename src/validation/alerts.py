# ==========================================================
# SISTEMA DE ALERTAS DE DATA VALIDATION
# ==========================================================

def generate_validation_alert(quality_status):
    # Genera un mensaje según el estado global de validación.
    status = quality_status.get("status")

    if status == "FAIL":
        return {
            "level": "ERROR",
            "message": "La validación de datos falló. El pipeline debe detenerse.",
        }

    if status == "WARNING":
        return {
            "level": "WARNING",
            "message": "La validación detectó advertencias que deben revisarse.",
        }

    return {
        "level": "INFO",
        "message": "La validación de datos fue completada correctamente.",
    }