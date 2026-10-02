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

- **Elemento de actuación**: la CNN-1D entrenada (`SeizureCNN`), que mapea cada ventana de entrada a un logit de crisis.
- **Crítica**: la función de pérdida (BCE con `pos_weight`) durante el entrenamiento, y las métricas clínicas (sensibilidad, especificidad, FPR, FP/hora) en validación y test, que juzgan la calidad de la acción según un criterio externo fijo.
- **Elemento de aprendizaje**: el optimizador AdamW con backpropagation, que modifica los pesos de la red para mejorar el desempeño futuro.
- **Generador de problemas**: el muestreo balanceado y aleatorizado del entrenamiento (submuestreo de la clase mayoritaria con `NEG_POS_RATIO`, shuffle de batches), que propone experiencias nuevas y variadas para explorar.

En la etapa de inferencia, el agente aprendido se comporta como un agente basado en utilidad: la salida sigmoide es una utilidad continua (probabilidad de crisis) que representa el grado de satisfacción con la hipótesis "hay crisis", y el umbral de decisión (`THRESHOLD`) permite el trade-off entre sensibilidad y especificidad. Subir el umbral prioriza la especificidad (menos falsas alarmas); bajarlo prioriza la sensibilidad (menos crisis perdidas), permitiendo una decisión racional bajo incertidumbre.

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

## 7. Experimentos y selección del modelo

Para comparar hiperparámetros sin mezclar pacientes entre entrenamiento y
validación se usa un **split inter-paciente fijo** (train/val/test), sin
validación cruzada (esto sería una mejora futura, lo ideal para la implementación de mi trabajo final de grado).

### 7.1 Split inter-paciente

`python -m src.split` genera `data/processed/split.json` con:

```text
train = 13 pacientes
val   = 5 pacientes
test  = 6 pacientes
```

La asignación es determinista para el mismo dataset y los mismos parámetros del
algoritmo y equilibra archivos y segundos de crisis. El val se
reserva de train y nunca se mezcla con él; el test queda congelado después de
generar el split.

El val incluye un paciente obligatorio de crisis cortas (`VAL_REQUIRED_PATIENTS`
= chb14) para que la selección del umbral "vea" el caso difícil. chb16 (crisis
de ~8 s) se mantiene en test.

### 7.2 Calcular el scaler (una sola vez)

Todos los escenarios comparten los mismos pacientes de train, así que el
RobustScaler se calcula una única vez:

```bash
python -m src.preprocessing --compute-stats \
    --data-dir <ruta/chbmit/1.0.0> \
    --stats-out data/processed/scaler_stats.npz
```

### 7.3 Entrenar cada escenario

Un escenario es una configuración fija de hiperparámetros. Se entrena una CNN
por escenario sobre los pacientes de train, con early stopping y selección del
punto de operación sobre val (regla de evento 2-de-3). Por ejemplo:

```bash
python -m src.train \
    --data-dir <ruta/chbmit/1.0.0> \
    --out models/control.pt \
    --pos-weight 1 --lr 1e-4 --weight-decay 1e-2 --dropout 0.4 \
    --noise-std 0.1 \
    --seed 42
```

La grilla de escenarios de referencia:

| escenario | pos_weight | lr    | weight_decay | dropout | hipótesis |
|-----------|------------|-------|--------------|---------|-----------|
| control   | 1.0        | 1e-4  | 1e-2         | 0.4     | reproducir baseline |
| pw3       | 3.0        | 1e-4  | 1e-2         | 0.4     | ponderación balanceada de positivas (crisis cortas) |
| lr_bajo   | 1.0        | 5e-5  | 1e-2         | 0.4     | estabilizar el entrenamiento |
| reg       | 1.0        | 1e-4  | 5e-2         | 0.5     | más L2 + dropout contra sobreajuste |

`--noise-std` controla el ruido gaussiano que se suma a las ventanas de crisis
(clase minoritaria) durante el train; `--noise-std 0` lo desactiva. `--dropout`
regula el dropout de la arquitectura.

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
