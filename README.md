# Detección Automática de Eventos Epilépticos en Señales EEG mediante CNN-1D

## 1. Definición del Problema
La inspección visual de registros de electroencefalogramas (EEG) resulta un proceso complejo y demandante para los neurólogos debido a la magnitud de los datos en monitoreos continuos. El objetivo es implementar una arquitectura de Inteligencia Artificial (Red Neuronal Convolucional o CNN 1D) capaz de clasificar ventanas temporales de señales crudas de EEG para diferenciar automáticamente entre períodos de crisis (actividad ictal) frente a períodos de actividad normal (basal).

### 1.1. Formulación del agente según el framework PEAS
Para especificar formalmente al agente, se utiliza el framework **PEAS** (*Performance, Environment, Actuators, Sensors*):

 **Performance (P)**  Clasificar correctamente cada ventana temporal de EEG como *crisis* (ictal) o *no crisis* (basal). Se cuantifica con métricas clínicas de clasificación binaria: **Sensibilidad** (Recall), **Especificidad**, **Tasa de Falsas Alarmas** (FPR) y **falsos positivos por hora** (FP/hora). El objetivo es maximizar la sensibilidad (no perder crisis) manteniendo el FPR y los FP/hora lo más bajos posibles, para que las alarmas sean clínicamente viables. 
 **Environment (E)**  El contexto clínico de monitoreo EEG continuo. Está constituido por la actividad eléctrica cerebral del paciente, captada en 16 canales bipolares (montaje TUEV) a 256 Hz, junto con el ruido y los artefactos propios del registro (musculares, oculares, de movimiento de electrodos, interferencia de red). La señal es continua y altamente no estacionaria. 
 **Actuators (A)**  El agente no actúa físicamente sobre el paciente: es un **sistema de apoyo a la decisión**. Su acción es emitir la clasificación binaria (crisis / no crisis) de cada ventana y, en un despliegue real, generar una alarma o aviso al personal médico cuando se detecta una crisis. 
 **Sensors (S)**  Los electrodos del montaje, que miden diferencias de potencial sobre el cuero cabelludo. La señal cruda de estos sensores, digitalizada a 256 Hz, filtrada en banda 0.5–50 Hz y segmentada en ventanas nominales de 5.12 s (1310 muestras, duración efectiva 5.117 s, con 50 % de solapamiento), es la percepción de entrada del modelo. 

#### 1.1.1. Características del entorno

 **Observabilidad:**  **Parcialmente observable.**  La señal útil está contaminada por ruido y artefactos que ocultan el estado real; la percepción es una ventana de señal y no el estado completo del cerebro. 
 **Determinismo:**  **Estocástico.**  La señal de EEG es aleatoria y no estacionaria: una misma condición clínica no produce una señal idéntica debido al ruido y los artefactos. El propio modelo introduce aleatoriedad (inicialización de pesos, dropout, batches estocásticos) y su salida es una probabilidad, no un valor determinista. 
 **Secuencialidad:**  **Episódico.**  Cada ventana se clasifica de forma independiente: la decisión sobre una ventana no condiciona las siguientes.
 **Dinamismo:**  **Estático.**  La actividad cerebral es dinámica, pero en el momento de la clasificación el agente no está interactuando con un entorno cambiante, sino que la inferencia opera sobre una ventana de señal congelada y la acción del agente no modifica el entorno.
 **Continuidad:**  **Continuo (entrada) / Discreto (salida).**  El estado de entrada (amplitudes de EEG) es continuo, de valores reales. La acción de salida es discreta y binaria (crisis / no crisis). El espacio continuo de probabilidades se discretiza aplicando un umbral de decisión.

 **Número de agentes:**  **Agente individual.**  Un único agente clasifica de forma autónoma, sin competir ni cooperar con otros agentes

#### 1.1.2. Tipo de programa de agente
El agente corresponde a un **agente que aprende**, entrenado mediante aprendizaje supervisado, y su arquitectura interna se corresponde con:

- **Elemento de actuación** (*performance element*): la CNN-1D entrenada (`SeizureCNN`), que mapea cada ventana de entrada a un logit de crisis.
- **Crítica** (*critic*): la función de pérdida (BCE con `pos_weight`) durante el entrenamiento, y las métricas clínicas (sensibilidad, especificidad, FPR, FP/hora) en validación y test, que juzgan la calidad de la acción según un criterio externo fijo.
- **Elemento de aprendizaje** (*learning element*): el optimizador **AdamW** con *backpropagation*, que modifica los pesos de la red para mejorar el desempeño futuro.
- **Generador de problemas** (*problem generator*): el muestreo balanceado y aleatorizado del entrenamiento (submuestreo de la clase mayoritaria con `NEG_POS_RATIO`, *shuffle* de batches), que propone experiencias nuevas y variadas para explorar.

En la etapa de inferencia, el agente aprendido se comporta además como un **agente basado en utilidad**: la salida sigmoide es una *utilidad* continua (probabilidad de crisis) que representa el grado de satisfacción con la hipótesis "hay crisis", y el umbral de decisión (`THRESHOLD`) permite el *trade-off* entre sensibilidad y especificidad. Subir el umbral prioriza la especificidad (menos falsas alarmas); bajarlo prioriza la sensibilidad (menos crisis perdidas), permitiendo una decisión racional bajo incertidumbre.

## 2. Origen y Naturaleza de los Datos
Se utiliza un dataset clínico de dominio público perteneciente al Hospital Infantil de Boston (CHB) en colaboración con el MIT, conformando el CHB-MIT Scalp EEG Database(https://physionet.org/content/chbmit/1.0.0/).
Estas son señales temporales continuas, altamente no estacionarias y afectadas por múltiples tipos de artefactos.
A nivel de preprocesamiento, se aplica un filtrado digital pasabanda de 0.5 a 50 Hz y una división en ventanas nominales de 5.12 segundos (1310 muestras, duración efectiva 5.117 s) con 50% de solapamiento. El pipeline procesa EDFs pregrabados de forma offline y usa `sosfiltfilt` para obtener un filtrado de fase cero.

El entorno requiere Python 3.11+ (las corridas actuales usan Python 3.14). Se fija una semilla aleatoria (seed) global para asegurar el determinismo en la división de lotes y entrenamiento.

## 3. Métricas de Éxito y Baseline
- Baseline Actual: Inspección visual manual (humana) que resulta en cargas analíticas excesivas y alta vulnerabilidad a la fatiga en monitoreos prolongados.
- Métricas de Éxito Clínico: Más allá de la exactitud global (Accuracy), la eficacia del sistema se medirá mediante:
- Sensibilidad (Recall): Capacidad de detectar correctamente las ventanas con crisis.
- Especificidad: Capacidad de excluir actividad de fondo o ruido.
- Tasa de Falsas Alarmas (FPR): Métrica fundamental para la viabilidad del sistema offline sobre registros pregrabados.

## 4. Límites del Alcance (para el MVP)
Quedan excluidos de esta primera entrega:
- Desarrollo de interfaces gráficas de usuario (Frontend web/apps).
- Implementación de bases de datos relacionales o históricos de pacientes.
- Transformación de las señales al dominio tiempo-frecuencia (Espectrogramas / Procesamiento 2D).
- Técnicas de Transfer Learning y preentrenamiento en datasets externos.

El MVP consistirá exclusivamente en un script por línea de comandos que recibe un bloque de datos temporales, realiza la inferencia utilizando la CNN-1D preentrenada, y retorna la clasificación binaria (Crisis / No Crisis).

## 5. Estructura del proyecto

```
ia-final/
├── README.md
├── requirements.txt
├── data/
│   └── processed/               <- split.json, ventanas generadas, stats
├── models/                      <- checkpoints (.pt) del modelo
├── notebooks/                   <- exploración
└── src/
    ├── __init__.py
    ├── config.py                <- configuración centralizada
    ├── annotations.py           <- parser de chbXX-summary.txt (anotaciones)
    ├── preprocessing.py         <- carga EDF, filtrado, ventaneo, etiquetado, escalado robusto (mediana + IQR)
    ├── data.py                  <- Dataset/DataLoader PyTorch
    ├── model.py                 <- arquitectura CNN-1D apilada
    ├── train.py                 <- loop de entrenamiento + métricas + checkpoint
    ├── evaluate.py              <- evaluación inter-paciente (val/test)
    ├── select_best.py           <- selección del mejor escenario por validación
    └── inference.py             <- CLI de inferencia sobre .edf
```

## 6. Instalación y ejecución (local)

### Requisitos
- Python 3.11+ (desarrollado y probado en 3.14).
- El dataset CHB-MIT descargado de PhysioNet:
  https://physionet.org/content/chbmit/1.0.0/

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Ruta del dataset

Todos los scripts aceptan `--data-dir` para apuntar a la carpeta del dataset (donde están las subcarpetas `chb01/`, `chb02/`, ...). 
Si no se pasa, se usa el default de `config.py`, que también se puede sobrescribir con la variable de entorno `CHBMIT_DIR`.

### Pipeline básico

```bash
# 1) Split inter-paciente (train/val/test) -> data/processed/split.json
python -m src.split --data-dir <ruta/chbmit/1.0.0>

# 2) Stats del escalador robusto -> data/processed/scaler_stats.npz
python -m src.preprocessing --compute-stats --data-dir <ruta/chbmit/1.0.0>

# 3) Entrenamiento -> models/best.pt
python -m src.train --data-dir <ruta/chbmit/1.0.0>

# 4) Evaluación sobre test (split no visto en entrenamiento)
python -m src.evaluate --checkpoint models/best.pt --split test --data-dir <ruta/chbmit/1.0.0>

# 5) Inferencia sobre un EDF suelto
python -m src.inference --input <archivo.edf> --checkpoint models/best.pt
```

## 7. Validación cruzada y experimentos

Para comparar hiperparámetros sin mezclar pacientes entre entrenamiento y
validación se usa validación cruzada agrupada por paciente.

El `split.json` actual mantiene el test fijo. Los pacientes de `train` y `val`
forman el conjunto de desarrollo y se dividen en cuatro folds. Cada fold tiene
12 pacientes para entrenamiento y 4 para validación. Nunca se mezclan ventanas
del mismo paciente entre ambos conjuntos.

El test no se usa para elegir hiperparámetros. Solo se evalúa una vez, cuando ya
se eligió la configuración final.

### 7.1 Generar los folds

```bash
python -m src.cross_validation \
    --split-file data/processed/split.json \
    --out data/processed/cv_splits.json \
    --n-folds 4 \
    --seed 42
```

Esto genera:

```text
data/processed/cv_splits.json
data/processed/cv_splits_fold_1.json
data/processed/cv_splits_fold_2.json
data/processed/cv_splits_fold_3.json
data/processed/cv_splits_fold_4.json
data/processed/cv_splits_final.json
```

### 7.2 Calcular el scaler de cada fold

Cada fold necesita sus propias estadísticas de RobustScaler. Las estadísticas se
calculan únicamente con los pacientes de train de ese fold.

```bash
python -m src.preprocessing --compute-stats \
    --data-dir <ruta/chbmit/1.0.0> \
    --split-file data/processed/cv_splits_fold_1.json \
    --stats-out data/processed/scaler_ratio3_pw1_fold_1.npz
```

El nombre incluye el experimento para no mezclar scalers entre configuraciones.
Se repite cambiando `fold_1` por `fold_2`, `fold_3` y `fold_4`.

### 7.3 Entrenar un experimento en los cuatro folds

Un experimento es una configuración fija de hiperparámetros. Por ejemplo,
`ratio3_pw1` significa `NEG_POS_RATIO=3` y `pos_weight=1`. Cada experimento
genera cuatro checkpoints, uno por fold. El ganador todavía no es uno de estos
checkpoints: es la configuración que funcione mejor en promedio.

```bash
python -m src.train \
    --data-dir <ruta/chbmit/1.0.0> \
    --split-file data/processed/cv_splits_fold_1.json \
    --scaler-stats data/processed/scaler_ratio3_pw1_fold_1.npz \
    --out models/cv/ratio3_pw1/fold_1.pt \
    --backup-dir <ruta/backup/ratio3_pw1/fold_1> \
    --neg-pos-ratio 3 \
    --pos-weight 1 \
    --seed 42
```

Para los siguientes folds se cambian las rutas y el nombre del checkpoint:

```text
models/cv/ratio3_pw1/fold_1.pt
models/cv/ratio3_pw1/fold_2.pt
models/cv/ratio3_pw1/fold_3.pt
models/cv/ratio3_pw1/fold_4.pt
```

Se repite cambiando `fold_1` por `fold_2`, `fold_3` y `fold_4`. El mismo
`pos_weight` y el mismo ratio deben mantenerse en los cuatro folds de un
experimento. Para probar otra configuración se crea otra carpeta, por ejemplo
`models/cv/ratio3_pw1.5/`.

En Colab, `--backup-dir` debe apuntar a una carpeta de Drive. Conviene usar una
carpeta diferente por experimento y fold para no sobrescribir checkpoints con el
mismo nombre.

### 7.4 Historiales de validación

Durante cada época, `src.train` evalúa automáticamente el fold de validación y
guarda las métricas en:

```text
models/cv/ratio3_pw1/fold_1.history.json
models/cv/ratio3_pw1/fold_1.oof.npz
models/cv/ratio3_pw1/fold_1.oof.json
```

Los archivos `.oof.*` son las predicciones del fold sobre sus pacientes de
validación y se usan después para fijar el threshold global. No se evalúa el test
durante esta etapa. `src.evaluate --split val` es opcional
si se quiere inspeccionar nuevamente un fold ya entrenado.

### 7.5 Comparar los experimentos

Después de terminar todos los folds de todos los experimentos se ejecuta:

```bash
python -m src.compare_cv \
    --root models/cv \
    --expected-folds 4 \
    --min-event-sensitivity 0.9 \
    --out results/cv_summary.json
```

El comparador agrupa los historiales por experimento y calcula:

- FDR promedio y desvío entre folds.
- Sensibilidad de evento promedio y desvío.
- Cantidad de folds completados.
- Elegibilidad según sensibilidad promedio `>= 0.90`.

La configuración elegida es la de menor FDR promedio entre las configuraciones
completas que alcanzan el objetivo de sensibilidad. Esta decisión se toma usando
validación, nunca usando test. El comparador también verifica que existan los
cuatro folds, sus checkpoints y sus archivos OOF, además de una configuración
consistente de ratio, `pos_weight`, seed y modo temporal. Un fold sin checkpoint
u OOF deja incompleto el experimento.

Por ejemplo, si imprime:

```text
Configuracion seleccionada por validacion: ratio3_pw1
```

significa que ganó la configuración `--neg-pos-ratio 3 --pos-weight 1`. No
significa que haya que elegir `fold_1.pt` como modelo final.

### 7.6 Fijar threshold y cantidad de épocas desde OOF

La configuración ganadora todavía necesita dos decisiones para el entrenamiento
final: el threshold y la cantidad de épocas. Se obtienen usando las predicciones
out-of-fold de los cuatro folds:

```bash
python -m src.finalize_cv \
    --experiment-dir models/cv/ratio3_pw1 \
    --expected-folds 4 \
    --min-event-sensitivity 0.9 \
    --out results/ratio3_pw1_recommendation.json
```

El archivo de recomendación contiene el threshold global elegido sobre todos los
pacientes OOF y la mediana de las mejores épocas de los folds.

### 7.7 Entrenamiento final del MVP

Una vez fijados configuración, threshold y épocas, se calcula un scaler con los
16 pacientes de desarrollo y se entrena un checkpoint nuevo sin validación
interna. La validación ya fue utilizada por la CV; el test continúa intacto.

Por ejemplo, si ganó `ratio3_pw1`:

```bash
python -m src.preprocessing --compute-stats \
    --data-dir <ruta/chbmit/1.0.0> \
    --split-file data/processed/cv_splits_final.json \
    --stats-out data/processed/scaler_final.npz

python -m src.train_final \
    --data-dir <ruta/chbmit/1.0.0> \
    --split-file data/processed/cv_splits_final.json \
    --scaler-stats data/processed/scaler_final.npz \
    --recommendation results/ratio3_pw1_recommendation.json \
    --out models/final.pt \
    --backup-dir <ruta/backup/final> \
    --seed 42
```

`models/final.pt` es un entrenamiento nuevo con los 16 pacientes de desarrollo.
No es una copia de ningún fold.

### 7.8 Evaluación final sobre test

Este es el único momento en que se usa el test para reportar el resultado final:

```bash
python -m src.evaluate \
    --checkpoint models/final.pt \
    --split test \
    --data-dir <ruta/chbmit/1.0.0> \
    --split-file data/processed/cv_splits_final.json
```

Después de esta evaluación no se deben cambiar hiperparámetros usando ese
resultado. Si se cambia algo, hay que repetir la selección sobre validación y
reservar el test nuevamente para el final.

### 7.6 Inferencia sobre un EDF

La inferencia no necesita labels. Puede ejecutarse sobre un EDF de train, val o
test para mostrar el funcionamiento del MVP, siempre que se aclare que usar un
EDF de test para una demostración no debe modificar el modelo ni sus
hiperparámetros.

```bash
python -m src.inference \
    --input <ruta/al/archivo.edf> \
    --checkpoint models/<ganador>.pt
```

La predicción de cada ventana se considera disponible al final de la ventana.
El informe de inferencia muestra tanto el inicio como el tiempo de decisión. La
inferencia procesa un EDF pregrabado completo; no es un sistema de streaming en
tiempo real.
