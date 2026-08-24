# Reporte de Data Quality — Air Quality

## 1. Objetivo
El objetivo de esta etapa es realizar un diagnóstico estructurado de la calidad de los datos almacenados en la capa RAW / Bronze antes de aplicar procesos de limpieza, transformación, análisis exploratorio de datos (EDA), Feature Engineering o entrenamiento de modelos.

El objetivo general del proyecto es seleccionar posteriormente una variable ambiental relevante y desarrollar un sistema de forecasting para pronosticar el comportamiento futuro de contaminantes o sensores.

La variable objetivo definitiva **no se selecciona durante Data Quality**. Su selección se realizará después del EDA, utilizando evidencia sobre calidad, cobertura, comportamiento temporal, estacionalidad, tendencia y relevancia de las variables.

## 2. Arquitectura de los datos
El proyecto utiliza SQL Server tanto para la fuente original como para la capa RAW / Bronze.

### Fuente
**dbo.AirQuality**
Esta tabla funciona como fuente original del proceso de ingesta.

### Proceso de ingesta
La ingesta es reproducible mediante:
bash python src/ingestion/ingest.py

El script obtiene los registros desde dbo.AirQuality y los carga en:
bronze.AirQuality

### Capa RAW / Bronze
**bronze.AirQuality**
Esta tabla constituye la capa RAW / Bronze utilizada por las etapas posteriores.
A partir de la finalización de la ingesta:
- Data Quality consulta **bronze.AirQuality**.
- Los Data Quality Gates se ejecutan sobre **bronze.AirQuality**.
- **dbo.AirQuality** no se utiliza directamente para realizar el diagnóstico de calidad.
- La capa Bronze conserva los datos ingeridos sin aplicar decisiones permanentes de limpieza destinadas al modelado.

Por lo tanto, el flujo utilizado es:
dbo.AirQuality
      │
      ▼
src/ingestion/ingest.py
      │
      ▼
bronze.AirQuality
      ├── Data Quality
      └── Data Quality Gates

## 3. Dimensiones de bronze.AirQuality
Después de ejecutar la ingesta se verificó:
- Registros en bronze.AirQuality: **9471**
- Columnas: **15**
- Registros completamente vacíos detectados: **114**

También se comprobó que la cantidad de registros de la fuente y de Bronze coincide después de la ingesta:
9471 → 9471
Esto permite verificar que la ingesta conserva la cantidad de registros esperada.

## 4. Columnas analizadas

bronze.AirQuality contiene:
- Date
- Time
- CO_GT
- PT08_S1_CO
- NMHC_GT
- C6H6_GT
- PT08_S2_NMHC
- NOx_GT
- PT08_S3_NOx
- NO2_GT
- PT08_S4_NO2
- PT08_S5_O3
- T
- RH
- AH
La capa Bronze mantiene la representación RAW de estas variables.

## 5. Valores faltantes

Se detectaron:
**114 registros completamente vacíos.**
Estos registros permanecen en bronze.AirQuality debido a que Bronze representa la capa RAW de los datos ingeridos.

Durante Data Quality no se aplicó  eliminación automática

La eliminación de registros durante esta etapa impediría analizar primero el impacto y la estructura de los valores faltantes.

Por lo tanto, su tratamiento se decidirá posteriormente durante Data Cleaning con base en los resultados del diagnóstico.

## 6. Valores faltantes codificados mediante -200

El dataset utiliza `-200` para representar ausencia de mediciones en diferentes variables.

Para realizar el diagnóstico se normalizó temporalmente la representación decimal y se interpretó `-200` como indicador de ausencia de medición.

Los resultados obtenidos sobre bronze.AirQuality fueron:

 Variable , Valores -200 , Porcentaje 

 CO_GT , 1683 , 17.77 % 
 PT08_S1_CO , 366 , 3.86 % 
 NMHC_GT , 8443 , 89.15 % 
 C6H6_GT , 366 , 3.86 % ,
 PT08_S2_NMHC , 366 , 3.86 % 
 NOx_GT , 1639 , 17.31 % 
 PT08_S3_NOx , 366 , 3.86 % 
 NO2_GT , 1642 , 17.34 % 
 PT08_S4_NO2 , 366 , 3.86 % 
 PT08_S5_O3 , 366 , 3.86 % 
 T , 366 , 3.86 % 
 RH , 366 , 3.86 % 
 AH , 366 , 3.86 % 

Estos valores no se eliminaron ni imputaron durante Data Quality.

## 7. Caso particular de NMHC_GT

NMHC_GT presenta el mayor problema de cobertura del dataset.

Se obtuvieron:
- Valores válidos: **914**
- Valores -200: **8443**
- Valores NULL: **114**
- Porcentaje -200: **89.15 %**
- Primer valor válido: **2004-03-10 18:00**
- Último valor válido: **2004-05-01 00:00**
Los datos válidos de esta variable se concentran únicamente en una parte inicial del período analizado.

## 8. Patrón temporal de valores faltantes

Se detectó que las siguientes nueve variables presentan exactamente **366 valores -200**:
- PT08_S1_CO
- C6H6_GT
- PT08_S2_NMHC
- PT08_S3_NOx
- PT08_S4_NO2
- PT08_S5_O3
- T
- RH
- AH
Se comprobó que:
- Filas con -200 simultáneamente en las nueve variables: **366**
- Filas con -200 en al menos una de ellas: **366**
- Los patrones corresponden exactamente a las mismas filas.
Por lo tanto, los missing de estas variables **no ocurren de forma independiente**.

## 9. Bloques temporales de ausencia de mediciones

Los 366 registros anteriores se analizaron cronológicamente y se identificaron **16 bloques temporales**.
Entre los más extensos:

| Inicio           | Final            | Registros |
| 2004-06-19 14:00 | 2004-06-21 03:00 | 38        |
| 2004-08-26 06:00 | 2004-08-28 02:00 | 45        |
| 2004-12-14 17:00 | 2004-12-17 19:00 | 75        |
| 2005-01-02 21:00 | 2005-01-05 00:00 | 52        |
| 2005-02-08 17:00 | 2005-02-11 20:00 | 76        |

### Interpretacion de los datos

Este resultado demuestra que una parte importante de los valores faltantes corresponde a **períodos continuos de ausencia de mediciones** y no simplemente a valores individuales distribuidos aleatoriamente.

Por esta razón, no se realizará interpolación automática de estos bloques durante Data Quality.

Interpolar períodos extensos podría generar una señal temporal artificial que los sensores realmente no observaron.

## 10. Duplicados

Se verificaron registros duplicados en bronze.AirQuality.
Resultado:
- Registros duplicados: **0**

## 11. Fechas, horas y continuidad temporal

Se analizaron las variables Date y Time.

Resultados:
- Fechas inválidas: **0**
- Horas inválidas: **0**
- Timestamps duplicados: **0**
- Horas faltantes en la secuencia RAW: **0**

### Interpretación

La estructura temporal de bronze.AirQuality mantiene la secuencia horaria esperada.

Sin embargo, es importante diferenciar entre:
1. timestamp inexistente;
2. timestamp existente con mediciones ausentes representadas mediante -200.

En este dataset se encontraron principalmente casos del segundo tipo.

Esta diferencia es especialmente importante porque el proyecto posteriormente desarrollará un sistema de forecasting.

## 12. Tipos de datos y convertibilidad numérica

Los datos almacenados en Bronze conservan su representación RAW.

Durante Data Quality se realizaron conversiones temporales mediante pandas.to_numeric() para comprobar la convertibilidad de las variables.

También se consideró la representación decimal mediante coma, por ejemplo:
2,6 → 2.6

Se comprobó que, después de normalizar correctamente la representación decimal, los valores pueden analizarse numéricamente considerando los NULL y -200 ya identificados.

### Nota

La conversión utilizada para el diagnóstico no modifica permanentemente bronze.AirQuality.
La transformación definitiva de tipos corresponderá a las etapas posteriores de preparación de datos.

## 13. Datos físicamente imposibles

Se realizaron comprobaciones iniciales sobre variables ambientales con límites físicos claramente interpretables.

### Temperatura

Rango observado después de excluir -200:
**-1.9 °C a 44.6 °C**

Valores fuera del intervalo de comprobación -50 °C a 60 °C:
**0**

### Humedad relativa

Rango observado:
**9.2 % a 88.7 %**

Valores fuera de 0 % a 100 %:
**0**

### Humedad absoluta

Rango observado:
**0.1847 a 2.231**

Valores negativos:
**0**

### Conclusión

No se identificaron valores físicamente imposibles bajo estas comprobaciones.

## 14. Valores extremos

Se utilizó el método IQR como herramienta de diagnóstico estadístico.
Resultados:

| Variable     | Outliers IQR |

| CO_GT        | 215          |
| PT08_S1_CO   | 118          |
| NMHC_GT      | 55           |
| C6H6_GT      | 228          |
| PT08_S2_NMHC | 65           |
| NOx_GT       | 435          |
| PT08_S3_NOx  | 241          |
| NO2_GT       | 107          |
| PT08_S4_NO2  | 97           |
| PT08_S5_O3   | 93           |
| T            | 3            |
| RH           | 0            |
| AH           | 2            |

### Justificación

Un outlier estadístico no representa necesariamente un dato incorrecto.
Por ejemplo, los tres outliers detectados para temperatura fueron:

| Fecha      | Hora  |Temperatura|

| 22/07/2004 | 15:00 | 44.3 °C   |
| 22/07/2004 | 16:00 | 44.6 °C   |
| 22/07/2004 | 17:00 | 43.4 °C   |

Los tres valores aparecen consecutivamente en el mismo período.
Esta coherencia temporal aporta evidencia para no considerarlos automáticamente errores de captura.

### conclusion

Los outliers **no serán eliminados automáticamente utilizando únicamente IQR**.
Su tratamiento deberá analizarse posteriormente durante EDA y dependerá de su comportamiento temporal y del modelo utilizado.

## 15. Cardinalidad

La cardinalidad observada fue:

| Variable     | Valores únicos |

| Date         | 391  |
| Time         | 24   |
| CO_GT        | 104  |
|PT08_S1_CO    | 1042 |
| NMHC_GT      | 430  |
| C6H6_GT      | 408  |
| PT08_S2_NMHC | 1246 |
| NOx_GT       | 926  |
| PT08_S3_NOx  | 1222 |
| NO2_GT       | 284  |
| PT08_S4_NO2  | 1604 |
| PT08_S5_O3   | 1744 |
| T            | 437  |
| RH           | 754  |
| AH           | 6684 |

### Interpretación

No se encontraron columnas constantes.

Time contiene exactamente **24 valores diferentes**, resultado consistente con una serie de mediciones horarias.
No se detectaron problemas evidentes de cardinalidad que requieran modificaciones.

## 16. Skewness

La asimetría se calculó excluyendo temporalmente -200.

| Variable     | Skewness |

| NOx_GT       | 1.7158 |
| NMHC_GT      | 1.5570 |
| CO_GT        | 1.3698 |
| C6H6_GT      | 1.3615 |
| PT08_S3_NOx  | 1.1017 |
| PT08_S1_CO   | 0.7559 |
| PT08_S5_O3   | 0.6279 |
| NO2_GT       | 0.6217 |
| PT08_S2_NMHC | 0.5616 |
| T            | 0.3094 |
| AH           | 0.2514 |
| PT08_S4_NO2  | 0.2054 |
| RH           | -0.0379|

### Interpretación

Algunos contaminantes presentan asimetría positiva considerable.
Esto puede estar relacionado con períodos de concentraciones elevadas y deberá investigarse durante EDA.

No se aplicarán transformaciones logarítmicas ni otras modificaciones de distribución durante Data Quality.

La necesidad de dichas transformaciones dependerá posteriormente de la variable seleccionada y del modelo de forecasting.

## 17. Correlación elevada

Se identificaron correlaciones absolutas iguales o superiores a 0.90.

| Variables              | Correlación |

| C6H6_GT — PT08_S2_NMHC | 0.9820 |
| CO_GT — C6H6_GT        | 0.9311 |
| CO_GT — PT08_S2_NMHC   | 0.9155 |
| NMHC_GT — C6H6_GT      | 0.9026 |

### Interpretación

La correlación elevada puede representar información redundante entre determinadas mediciones y sensores.

### Decisión

No se eliminarán variables únicamente por presentar correlación elevada.
Durante EDA y Feature Engineering se determinará si estas relaciones aportan información útil o producen redundancia/multicolinealidad para el modelo seleccionado.

## 18. Diagnóstico preliminar orientado a forecasting

El objetivo final del proyecto es:
**Seleccionar una variable ambiental relevante y desarrollar un sistema de forecasting para pronosticar el comportamiento de contaminantes o sensores.**

Durante Data Quality se analizaron diferentes variables ambientales como **candidatas**, sin seleccionar todavía una variable objetivo definitiva.

La cobertura observada fue:

| Variable | Valores válidos | Faltantes | Cobertura |

| CO_GT    | 7674            | 1797      | 81.03 % |
| NMHC_GT  | 914             | 8557      | 9.65 % |
| C6H6_GT  | 8991            | 480       | 94.93 % |
| NOx_GT   | 7718            | 1753      | 81.49 % |
| NO2_GT   | 7715            | 1756      | 81.46 % |

También se realizó un diagnóstico preliminar de autocorrelación:

| Variable | Lag 1h | Lag 24h|Lag 168h|

| CO_GT    | 0.8370 | 0.6045 | 0.5608 |
| C6H6_GT  | 0.8394 | 0.6328 | 0.5988 |
| NOx_GT   | 0.9030 | 0.6741 | 0.5984 |
| NO2_GT   | 0.8995 | 0.7072 | 0.6465 |

### Importante

Estos resultados **NO representan la selección de la variable objetivo**.

Su propósito es demostrar que existen variables con suficiente estructura temporal para continuar posteriormente con un análisis exploratorio orientado a forecasting.

La selección definitiva se realizará después del EDA considerando conjuntamente:
- cobertura;
- calidad de datos;
- distribución;
- tendencia;
- estacionalidad;
- autocorrelación;
- comportamiento temporal;
- relevancia ambiental;
- relación con otras variables;
- viabilidad de tratamiento de valores faltantes;
- desempeño potencial para forecasting.

## 19. Evidencia temporal preliminar

Como parte del diagnóstico se profundizó en una de las variables candidatas para comprobar si el dataset contiene patrones temporales aprovechables.

Se observaron autocorrelaciones significativas en diferentes horizontes y diferencias entre horas del día.
Este análisis se considera **exploratorio y preliminar**.
No implica que dicha variable haya sido seleccionada como target.
El EDA deberá aplicar una comparación sistemática entre las variables candidatas antes de tomar la decisión definitiva.

## 20. Leakage

El proyecto final corresponde a un problema de forecasting.
Por esta razón, el leakage deberá analizarse principalmente desde una perspectiva temporal.
Durante las etapas posteriores deberán cumplirse las siguientes condiciones:
- Train, validation y test deberán respetar el orden cronológico.
- No se utilizará una división aleatoria que permita que información futura termine en entrenamiento.
- Los lags deberán utilizar únicamente observaciones anteriores al instante pronosticado.
- Las estadísticas móviles deberán calcularse sin incorporar observaciones futuras.
- Los procesos que aprendan parámetros de los datos deberán ajustarse utilizando únicamente el conjunto correspondiente de entrenamiento cuando sea necesario.
- La creación de features no podrá utilizar información que no estaría disponible en el instante real de predicción.

### Estado actual

En Data Quality todavía no existe el pipeline definitivo de features ni la partición train/validation/test.
Por lo tanto, no corresponde afirmar todavía que el proyecto está completamente libre de leakage.
La prevención deberá verificarse nuevamente cuando se implemente el pipeline de forecasting.

## 21. Imbalance

El objetivo del proyecto es forecasting de una variable ambiental continua.
No corresponde a un problema tradicional de clasificación con clases discretas.
Por esta razón, **class imbalance no aplica directamente** en esta etapa.
Posteriormente sí deberá analizarse la distribución de la variable seleccionada, los períodos de valores altos/bajos y la representación temporal de los diferentes comportamientos.

## 22. Categorías inconsistentes

El dataset contiene principalmente:
- variables temporales;
- contaminantes;
- respuestas de sensores;
- variables meteorológicas.
No se identificaron variables categóricas de negocio que requieran normalización de etiquetas o categorías.
Por lo tanto, el diagnóstico tradicional de categorías inconsistentes **no aplica directamente** al dataset actual.

## 23. Errores de unidad

Se realizaron comprobaciones de coherencia mediante rangos y análisis de las variables ambientales.
No se encontró durante Data Quality evidencia suficiente para afirmar la existencia de errores o cambios de unidad dentro de una misma variable.

Sin embargo, la interpretación de las unidades deberá mantenerse de acuerdo con la documentación original del dataset.

Durante EDA se deberá considerar la escala y unidad correspondiente antes de comparar variables diferentes.

## 24. Anomalías estadísticas

El análisis identificó diferentes comportamientos que requieren consideración:
- valores extremos detectados mediante IQR;
- asimetría positiva en determinados contaminantes;
- correlaciones elevadas;
- períodos continuos de ausencia de mediciones;
- una variable con ausencia extremadamente elevada (NMHC_GT).

Estos hallazgos se consideran anomalías o características estadísticas que requieren análisis, pero **no se clasifican automáticamente como errores**.
Las decisiones de tratamiento se realizarán posteriormente con base en EDA y Data Cleaning.

## 25. Decisiones tomadas durante Data Quality

A partir del diagnóstico se establecieron las siguientes decisiones:

1. Utilizar bronze.AirQuality como dataset RAW para Data Quality.
2. Mantener dbo.AirQuality únicamente como fuente del proceso de ingesta.
3. No modificar permanentemente la capa Bronze durante Data Quality.
4. No eliminar automáticamente las 114 filas completamente vacías.
5. Interpretar -200 como ausencia de medición durante el diagnóstico.
6. No imputar automáticamente los bloques temporales largos.
7. No realizar una imputación masiva de NMHC_GT.
8. No eliminar automáticamente outliers identificados mediante IQR.
9. No eliminar variables únicamente por correlación elevada.
10. No aplicar transformaciones de distribución únicamente por skewness.
11. Mantener el orden temporal como requisito para las etapas posteriores.
12. No seleccionar todavía la variable objetivo.
13. Realizar la selección de la variable ambiental después del EDA.
14. Justificar durante Data Cleaning cada eliminación, imputación o transformación que se realice.

## 26. Data Quality Gates

El proyecto implementa validaciones automáticas estructuradas antes de continuar hacia etapas posteriores.

Actualmente se comprueban:
1. Dataset no vacío.
2. Presencia de las columnas esperadas.
3. Registros duplicados.
4. Filas completamente vacías.
5. Fechas válidas.
6. Horas válidas.
7. Convertibilidad de variables numéricas.
8. Valores faltantes codificados.
9. Continuidad temporal.

Los resultados son procesados mediante:

quality_checks.py → quality_gates.py → alerts.py

El sistema utiliza estados:
- PASS
- WARNING
- FAIL

## 27. Resultado actual de los Data Quality Gates

La ejecución actual produce:
WARNING: La alerta generada es:
La validación detectó advertencias que deben revisarse.

### Interpretación

El estado WARNING es coherente con el diagnóstico.
El dataset presenta problemas que deberán tratarse posteriormente, pero las validaciones realizadas hasta el momento no indican que la estructura completa del dataset sea inutilizable.


## 28. Conclusión

El diagnóstico realizado sobre bronze.AirQuality demuestra que el dataset conserva una estructura temporal adecuada para continuar hacia las siguientes etapas, pero presenta diferentes problemas de calidad que deben ser tratados explícitamente.

Los principales hallazgos son:

- 9471 registros en Bronze;
- 114 registros completamente vacíos;
- valores faltantes codificados mediante -200;
- ausencia extremadamente elevada en `NMHC_GT`;
- 366 ausencias simultáneas en diferentes sensores y variables ambientales;
- 16 bloques temporales de ausencia de mediciones;
- ausencia de registros duplicados;
- fechas y horas válidas;
- continuidad de timestamps en la estructura RAW;
- valores extremos que no deben considerarse automáticamente errores;
- diferentes niveles de skewness;
- correlaciones elevadas entre algunas variables;
- evidencia de estructura temporal útil para forecasting.

Durante Data Quality **no se modificó permanentemente la capa bronze.AirQuality y no se seleccionó una variable objetivo definitiva**.
Los resultados obtenidos servirán como fundamento para las decisiones posteriores de:
Data Cleaning → EDA → selección de variable → Feature Engineering → forecasting
La variable ambiental que se pronosticará será seleccionada formalmente después del EDA utilizando evidencia y no únicamente cobertura o autocorrelación.