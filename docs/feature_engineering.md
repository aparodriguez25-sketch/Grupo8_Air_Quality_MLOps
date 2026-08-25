# Feature Engineering

## 1. Objetivo

La etapa de Feature Engineering transforma los datos preparados durante Data Cleaning en variables predictivas que puedan ser utilizadas posteriormente por los modelos de forecasting.

La variable ambiental seleccionada durante el EDA es:

C6H6_GT — concentración de benceno.

El objetivo predictivo definido es:

**predecir la concentración de benceno una hora hacia el futuro utilizando únicamente información disponible hasta la hora actual.**

## 2. Principio de reutilización

Las transformaciones de Feature Engineering se implementan como código reutilizable dentro de:

src/features/

La estructura utilizada es:

- transformations.py: contiene las transformaciones, validaciones y definición de features.
- build_features.py: ejecuta el pipeline completo de construcción de features.
- __init__.py: define el directorio como módulo de Python.

El notebook no implementará una lógica independiente de Feature Engineering.

La misma implementación deberá reutilizarse durante:

- experimentación;
- entrenamiento;
- evaluación;
- inferencia;
- producción.

Esto evita mantener una lógica de **Notebook Feature Engineering** diferente de **Production Feature Engineering**.

## 3. Validación temporal

Antes de generar variables predictivas se valida la estructura temporal del dataset.

El pipeline comprueba:

- existencia de la columna timestamp;
- conversión correcta a formato datetime;
- orden cronológico;
- ausencia de timestamps duplicados;
- frecuencia horaria continua.

Después de Data Cleaning se obtuvieron:

- 9,357 observaciones;
- 9,356 intervalos consecutivos;
- todos los intervalos corresponden exactamente a una hora.

Por lo tanto, las operaciones basadas en shift() representan correctamente rezagos temporales horarios.

## 4. Variables temporales

A partir de `timestamp` se generan:

- hour: hora del día;
- day_of_week: día de la semana;
- month: mes.

Estas variables se justifican por los patrones horarios, semanales y mensuales identificados durante el EDA.

## 5. Variables de rezago

Para representar la dependencia temporal de C6H6_GT se generan:

- lag_1: concentración una hora antes;
- lag_2: concentración dos horas antes;
- lag_3: concentración tres horas antes;
- lag_24: concentración 24 horas antes;
- lag_168: concentración 168 horas antes, equivalente a siete días.

El EDA mostró una autocorrelación aproximada de:

- 0.839 a 1 hora;
- 0.632 a 24 horas.

Estos resultados justifican incorporar información histórica como variables predictivas.

## 6. Estadísticas móviles

Se generan las siguientes variables:

- rolling_mean_3: media de las tres horas anteriores;
- rolling_mean_24: media de las 24 horas anteriores;
- rolling_std_24: desviación estándar de las 24 horas anteriores.

Antes de calcular las ventanas móviles se utiliza:

shift(1)

Esto garantiza que el valor actual de C6H6_GT no participe en la construcción de una feature histórica.

Por ejemplo, para predecir una concentración futura desde las 15:00, una ventana móvil puede utilizar información disponible hasta las 15:00 o anterior según la definición del predictor, pero nunca información de horas posteriores.

## 7. Prevención de Data Leakage

Todas las variables relacionadas con el comportamiento histórico de C6H6_GT utilizan únicamente información disponible hasta el instante actual.

No se utilizan observaciones futuras para construir:

- lags;
- medias móviles;
- desviaciones móviles;
- variables temporales.

La separación posterior de Train, Validation y Test también deberá respetar estrictamente el orden cronológico.

## 8. Target de forecasting

Para convertir el problema en una predicción real hacia el futuro se creó:

target_next_hour

Esta variable representa el valor de C6H6_GT una hora después del instante actual.

Se genera mediante un desplazamiento de una hora hacia el futuro del target original.

Ejemplo:

Hora actual         C6H6_GT     target_next_hour

18:00                 11.9              9.4
19:00                  9.4              9.0
20:00                  9.0              9.2
21:00                  9.2              6.5

---

## 9. Uso de C6H6_GT actual como predictor

C6H6_GT en el instante actual se mantiene como una feature predictiva.

Esto no constituye Data Leakage porque el objetivo del modelo es predecir:
C6H6_GT(t + 1)
utilizando únicamente información disponible hasta:t

Por lo tanto, la concentración conocida en la hora actual puede utilizarse para predecir la concentración de la siguiente hora.

## 10. Tratamiento de valores faltantes

Después de Data Cleaning, `C6H6_GT` presentaba:

- 366 valores faltantes.

Para evitar utilizar información futura, se implementó una estrategia de imputación causal dentro del pipeline reutilizable de Feature Engineering.

Se utiliza: ffill(limit=3)

Esta estrategia rellena únicamente huecos cortos de hasta tres horas utilizando el último valor conocido.

La imputación se aplica a C6H6_GT cuando actúa como predictor.

El target: target_next_hour

se crea antes de realizar esta imputación. Por esta razón, el valor objetivo siempre corresponde a una medición real observada y nunca a un valor imputado.

La estrategia cumple los siguientes criterios:

- no se imputa target_next_hour;
- no se utiliza información futura;
- no se utiliza backfill;
- únicamente se rellenan hasta tres horas consecutivas;
- los bloques largos permanecen como valores faltantes;
- los lags se calculan después de la imputación causal;
- las estadísticas móviles se calculan después de la imputación causal.

Después de aplicar la estrategia, los valores faltantes de C6H6_GT utilizado como predictor disminuyeron:

- Antes: 366.
- Después: 324.

Por lo tanto, se recuperaron 42 observaciones del predictor sin rellenar artificialmente los bloques largos.

## 11. Evaluación de rolling_mean_168

Durante Feature Engineering se evaluó inicialmente: rolling_mean_168

Esta variable representaba la concentración promedio de las 168 horas anteriores.

Sin embargo, al exigir 168 observaciones válidas consecutivas, los valores faltantes originales afectaban una gran cantidad de ventanas.

Con rolling_mean_168 se obtenían:

- 6,303 observaciones completamente utilizables;
- 3,054 observaciones excluidas;
- 67.36 % de cobertura.

La pérdida de información fue considerada demasiado alta.

Por esta razón, rolling_mean_168 fue descartada.

Se conservó:

lag_168

para representar el comportamiento semanal con una menor pérdida de observaciones.

## 12. Features seleccionadas

Para el primer modelo se definieron 12 features predictivas:

- C6H6_GT, hour, ay_of_week, month, lag_1
- lag_2, lag_3, lag_24, lag_168, rolling_mean_3
- rolling_mean_24, rolling_std_24

La variable objetivo es: target_next_hour

La selección se encuentra centralizada mediante FEATURE_COLUMNS y MODEL_TARGET_COLUMN para evitar mantener listas diferentes entre experimentación y producción.

## 13. Variables no utilizadas inicialmente

Las siguientes variables originales no forman parte del primer conjunto predictivo:

- CO_GT, PT08_S1_CO, NMHC_GT, PT08_S2_NMHC, NOx_GT
- PT08_S3_NOx, NO2_GT, PT08_S4_NO2, PT08_S5_O3, T, RH, AH

Estas variables podrán evaluarse posteriormente mediante experimentos para determinar si mejoran el desempeño predictivo.

Date y Time tampoco se utilizan directamente como predictores porque su información se representa mediante las features temporales derivadas.

timestamp se conserva para mantener el orden cronológico y realizar posteriormente la separación Train, Validation y Test.

## 14. Dataset resultante de Feature Engineering

Después de ejecutar el pipeline de Feature Engineering se obtiene:

- 9,357 filas;
- 28 columnas.

La arquitectura implementada es:

Data Cleaning
      ↓
Validación temporal
      ↓
Creación de target t+1
      ↓
Imputación causal de huecos cortos
      ↓
Features temporales
      ↓
Lags
      ↓
Rolling features
      ↓
Dataset de Feature Engineering