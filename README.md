<div align="center">

# Detección Automática de Eventos Epilépticos en Señales EEG mediante CNN-1D

### Clasificación automática de ventanas de EEG para distinguir crisis epilépticas de actividad basal, como apoyo al monitoreo clínico continuo

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

---

## Sobre el proyecto

Inspeccionar visualmente un registro de electroencefalograma (EEG) es una tarea lenta y demandante para los neurólogos: en monitoreos continuos se acumulan horas de señal por paciente, y la revisión manual no escala ni está exenta de fatiga. Este proyecto entrena un clasificador automático que distingue, ventana a ventana, los períodos de crisis (actividad ictal) de la actividad basal (normal).

El enfoque elegido es una red neuronal convolucional 1D (CNN-1D) que opera directamente sobre la señal cruda filtrada, sin necesidad de transformaciones manuales a otros dominios. El sistema se entrena con aprendizaje supervisado y evalúa su desempeño con métricas clínicas de detección de crisis (sensibilidad, falsas alarmas por hora y latencia de detección), no solo con accuracy.

El resultado principal es que la CNN-1D alcanza una sensibilidad de crisis del 67,6 % (23 de 34 crisis detectadas) sobre pacientes no vistos, con un AUPRC de 0,39, superando de forma consistente a los baselines clásicos (el clasificador mayoritario y modelos de machine learning sobre potencia por banda). El modelo es un primer MVP de apoyo a la decisión, todavía por debajo de los objetivos clínicos de sensibilidad y de tasa de falsas alarmas.

### Contexto académico

- Asignatura: Inteligencia Artificial
- Institución: Instituto Universitario Aeronáutico
- Cohorte / año: 2026
- Docente: Gonzalez, Hernando Alexis

### Construido con

- [Python 3.11+](https://www.python.org/) (desarrollado y probado en 3.14)
- [PyTorch](https://pytorch.org/) / [torchinfo](https://github.com/TylerYep/torchinfo) — red neuronal y resúmenes de arquitectura
- [MNE](https://mne.tools/) / [SciPy](https://scipy.org/) — carga de EDF, filtrado y procesamiento de señal
- [NumPy](https://numpy.org/) / [scikit-learn](https://scikit-learn.org/) — datos y métricas
- [matplotlib](https://matplotlib.org/) — gráficos de resultados
- [FastAPI](https://fastapi.tiangolo.com/) / [Vue 3](https://vuejs.org/) / [ECharts](https://echarts.apache.org/) — demo interactiva de inferencia
- [pip](https://pip.pypa.io/) / [venv](https://docs.python.org/3/library/venv.html) — gestión de dependencias y entorno

---

## El dataset

Se usa el CHB-MIT Scalp EEG Database, un dataset clínico de dominio público del Hospital Infantil de Boston (CHB) en colaboración con el MIT. Son señales temporales continuas, no estacionarias y afectadas por distintos tipos de artefactos (musculares, oculares, de movimiento de electrodos, interferencia de red).

> El dataset crudo pesa ~64 GB y no se incluye en este repositorio. Se descarga de PhysioNet y se apunta con `--data-dir` (o la variable de entorno `CHBMIT_DIR`). Los datos ya procesados (split y estadísticas del escalador) sí están versionados en `data/processed/`.

| Campo | Detalle |
|---|---|
| Fuente | [CHB-MIT Scalp EEG Database (PhysioNet)](https://physionet.org/content/chbmit/1.0.0/) |
| Registros | 24 pacientes (chb01–chb24), EEG continuo a 256 Hz |
| Variables | 16 canales bipolares (montaje TUEV) + anotaciones de crisis (`chbXX-summary.txt`) |
| Variable objetivo | Crisis (ictal) vs. no crisis (basal), a nivel de ventana de 5,12 s |
| Período / cobertura | Monitoreo continuo pediátrico, múltiples sesiones por paciente |
| Licencia del dataset | Público (PhysioNet); uso académico. No redistribuido aquí |

---

## Metodología

El pipeline completo es: carga del EDF → filtrado → segmentación en ventanas → escalado → entrenamiento de la CNN → selección del punto de operación → evaluación inter-paciente. Todo está implementado como módulos ejecutables (`python -m src.*`) en `src/`.

### Análisis exploratorio

Las señales EEG son continuas, no estacionarias y con fuerte desbalance entre clases: las crisis representan una fracción mínima del tiempo total de registro. Los artefactos y el ruido contaminan la señal útil, y la duración de las crisis varía mucho entre pacientes (hay crisis muy cortas, de ~8 s, que son las más difíciles). Este desbalance y la variabilidad inter-paciente motivan las decisiones de diseño siguientes.

### Preprocesamiento

- Filtrado pasabanda 0,5–50 Hz con `sosfiltfilt` (fase cero), orden 5.
- Segmentación en ventanas nominales de 5,12 s (1310 muestras a 256 Hz) con 50 % de solapamiento, siguiendo el montaje TUEV de 16 canales.
- Escalado robusto por canal (mediana e IQR, `RobustScaler`), calculado solo sobre los pacientes de entrenamiento para evitar fuga de datos.
- Balanceo mediante submuestreo de la clase mayoritaria (`NEG_POS_RATIO = 3`) y augmentación con ruido gaussiano solo en ventanas de crisis.

### Modelo(s)
TODO: CAMBIAR ESTO PQ ABAJO HABLA DE SPLIT FIJO
Se entrenó una CNN-1D de 6 bloques (Conv1d + BatchNorm + ReLU + MaxPool) seguidos de Global Average Pooling y una capa densa. Se compararon dos funciones de pérdida:

| Configuración | Pérdida | AUPRC (val) | Sens. evento (val) | Falsas alarmas/h (mediana, val) |
|---|---|---|---|---|
| BCE (baseline del entrenamiento) | BCE con `pos_weight` | 0,52 | 0,811 (30/37) | 0,42 |
| Focal `alpha=0.25` | Focal Loss (Lin et al.) | 0,55 | 0,838 (31/37) | 0,42 |
| Focal `alpha=0.75` | Focal Loss | 0,58 | 0,757 (28/37) | 0,20 |

La Focal Loss pondera menos los ejemplos fáciles y enfoca el entrenamiento en las crisis cortas y los falsos positivos, que son los casos difíciles.
El checkpoint se elige por el *punto de operación clínico* (máxima sensibilidad de evento sujeta a una mediana de falsas alarmas ≤ 1/h por paciente), no por AUPRC, el cual se usa para desempatar. En validación, `alpha=0.75` ordena mejor (AUPRC 0,58) y dispara menos falsas alarmas, pero detecta ~3 crisis menos (28/37 vs 31/37), y el objetivo prioriza no perder crisis.

Validación: split inter-paciente fijo (train/val/test, sin mezclar ventanas del mismo paciente entre particiones) para evitar fuga de datos. El punto de operación (umbral) se elige en validación maximizando la sensibilidad de crisis sujeta a un techo de falsas alarmas (mediana ≤ 1/h por paciente), con postprocesado a nivel de evento (≥2 positivas en 3 ventanas consecutivas, intervalo mínimo de 4 min entre alarmas).

### Validación cruzada de desarrollo

Se implementó una validación cruzada agrupada por paciente para evaluar las configuraciones de entrenamiento sobre distintas particiones del dataset de desarrollo (train y val, 18 pacientes). Los datos se dividen en K folds, y en cada iteración se entrena el modelo con los pacientes de (K−1) folds y se evalúa con los pacientes del fold restante. Cada fold produce una medida de desempeño y los resultados se resumen mediante la media y el desvío estándar.

La partición se realiza por paciente, no por ventana, para mantener todas las ventanas de cada paciente en un mismo grupo. Se utiliza StratifiedGroupKFold con K=5 y estratificación por severidad de crisis. Los 18 pacientes del pool de desarrollo rotan y aparecen una vez cada uno en validación. El conjunto de test del split fijo se mantiene separado durante la validación cruzada.

En cada fold, las estadísticas del RobustScaler se calculan únicamente con los pacientes de entrenamiento de ese fold. Se comparan las configuraciones `bce`, `focal_a025` y `focal_a075`, manteniendo constantes los demás parámetros del entrenamiento. Para cada configuración se ejecutan cinco entrenamientos, uno por fold, y se reportan AUPRC, AUC-ROC, métricas por ventana y por evento, falsas alarmas por hora y latencia, agregadas entre folds.

La configuración se selecciona según la media de AUPRC de validación; en caso de empate, se prefiere la de menor desvío estándar. Luego se reentrena esa configuración con todo el pool de desarrollo y se evalúa una vez sobre el conjunto de test.
---

## Resultados y comparación contra baseline

Métricas sobre el test congelado inter-paciente (6 pacientes, 34 crisis). Los baselines de comparación son dos modelos de machine learning sobre features
de ingeniería (potencia espectral de cada ventana en las bandas canónicas del EEG), entrenados como regresión logística y SVM lineal. Se usa el mismo split inter-paciente, el mismo postprocesado de evento y el mismo criterio de punto de operación que la CNN. 

| Métrica  | Regresión logística |SVM lineal | BCE | Focal α=.25 (final) |
|----------|---------------------|-----------|-----|---------------------|
| AUC-ROC | 0,694 | 0,710 | 0,8915 | 0,8957 |
| AUPRC   | 0,030 | 0,064 |0,3736 | 0,3881 |
| Sensibilidad de crisis (evento) | 26,5 % (9/34) | 17,6% (6/34) | 64,7 % (22/34) | 67,6 % (23/34) |
| Falsas alarmas/h (mediana por paciente) | 12,55 | 0,41 | 2,05 | 2,42 |
| Sensibilidad (ventana)                  | 0,593 | 0,189 | 67,3 % | 63,6 % |
| Especificidad (ventana)                 | 69,6% | 99,6% | 95,7 % | 97,2 % |
| MCC (ventana)                           | - | - | 0,139 | 0,164 |
| Latencia media de detección | 14,9s | 17,6s |16,70 s | 15,85 s |

Focal α=.25 es superior a BCE en ranking (AUPRC y AUC) y en la mayoría de las métricas de ventana: reduce los falsos positivos (~36 % menos de FPR) y mejora MCC y F1.
En comparación con los baselines, la CNN supera ampliamente a ambos: mejora el AUC-ROC de ~0,71 a 0,90, multiplica el AUPRC por más de 6 (0,39 vs 0,06), detecta más del doble de crisis (23/34 vs 9/34 y 6/34). El SVM lineal logra respetar el techo clínico de falsas alarmas (0,41/h) pero a costa de una sensibilidad muy baja (17,6 %).

### Análisis crítico

El modelo es un MVP funcional que supera al baseline, pero no alcanza todavía los objetivos clínicos:

- La sensibilidad de crisis (67,6 %) queda lejos del objetivo de ≥90 %.
- Las falsas alarmas (~2,4/h medianas) superan el techo clínico de 1/h.
- Las crisis cortas siguen siendo el punto débil: en `chb16` (crisis de ~8 s) solo se
  detectan 2 de 10 para ambos modelos. Es un problema de representación de datos, no de la
  función de pérdida.
- Con 34 crisis de test, la mejora de una crisis entre BCE y Focal tiene aún mucha
  incertidumbre estadística.

---

## Estructura del proyecto

```text
.
├── README.md
├── pyproject.toml
├── LICENSE
├── INFORME_TECNICO.pdf    # informe técnico del trabajo
├── src/                   # código fuente (funciones y módulos propios)
│   ├── api.py             # API FastAPI + demo de inferencia
│   ├── baseline.py        # baselines clásicos (potencia por banda: LR y SVM lineal)
│   └── cv.py              # validación cruzada agrupada por paciente
├── frontend/              # interfaz web de la demo (Vue 3 + ECharts)
├── data/
│   ├── raw/               # CHB-MIT crudo (~64 GB, NO versionado)
│   └── processed/         # split.json y scaler_stats.npz
├── models/                # checkpoints entrenados (se publican como Release, NO versionados)
├── inferences/            # reportes JSON de inferencia (ejemplos)
└── tests/                 # tests de unidad (pytest)
```

---

## Instalación

### Prerrequisitos

- Git
- [Node.js](https://nodejs.org/) (para compilar la interfaz web)
- El dataset CHB-MIT de [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) (~64 GB, para re-entrenar o evaluar. Para inferencia no hace falta).

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/agostinaperez/ia-final.git
cd ia-final

# 2. Crear el entorno virtual y activarlo (Python 3.11+)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 3. Instalar las dependencias (el paquete `src` se instala en modo editable)
pip install -e .

# 4. Descargar el checkpoint de la última Release y dejarlo en models/
#    (ej.: https://github.com/agostinaperez/ia-final/releases -> alpha025.pt)
```

El dataset solo se necesita para el pipeline completo. En ese caso, setear su ruta en `.env` (`CHBMIT_DIR`, a partir de `.env.example`) o pasar `--data-dir <ruta>` a cada script.

> Nota CPU/CUDA: `pyproject.toml` fija `torch==2.14.0`. Si la máquina no tiene GPU, instalar
> explícitamente el wheel CPU: `pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu`.
---

## Uso

```bash
# Split inter-paciente (train/val/test) -> data/processed/split.json
python -m src.split --data-dir <ruta/chbmit/1.0.0>

# 5 folds agrupados por paciente sobre el pool de desarrollo (18 pacientes). Salidas en results/cv/. Incluye la evaluación con split de test
python -m src.cv --data-dir <ruta/chbmit/1.0.0> --folds 5 --seed 42

Costo: son 15 entrenamientos completos (3 configs × 5 folds), pensados para GPU (Colab). Cada época guarda en `--out-dir` un estado reanudable (`.resume.pt`) y el mejor checkpoint del fold; al completar un fold, también persiste sus métricas. 

# Baselines clásicos (regresión logística / SVM lineal sobre potencia por banda)
python -m src.baseline --data-dir <ruta/chbmit/1.0.0> --model all
```

Para inferir sobre un EDF suelto no se necesitan los 64 GB ni `CHBMIT_DIR`: las estadísticas del escalador van embebidas en el checkpoint.

```bash
# Informe ventana a ventana + eventos detectados (JSON en inferences/)
python -m src.inference --input <archivo.edf> --checkpoint models/alpha025.pt
```

### Inferencia con interfaz web.

```bash
cd frontend
npm install
npm run build
cd ..

# 2. Levantar la API y servir la interfaz (Vue + ECharts)
python -m src.api
# o, si se prefiere uvicorn directamente:
uvicorn src.api:app --host 127.0.0.1 --port 8000
```
Abrir `http://127.0.0.1:8000` en el navegador. La interfaz permite:

- Elegir un modelo de entre los checkpoints `.pt` de `models/` y ver su historial de
  entrenamiento (curvas de loss y métricas), la info de cómputo (parámetros, MACs/FLOPs,
  latencia) y su configuración.
- Subir un EDF (con el selector de archivos local), correr la inferencia y ver la curva
  de probabilidad de crisis por ventana, la señal EEG con los eventos detectados resaltados
  y el detalle de cada evento (inicio, fin, duración).

---

## Tests

Tests de unidad que no requieren el dataset (preprocesamiento, métricas, modelo y API):

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## Autora

- Pérez López Agostina - https://github.com/agostinaperez - https://www.linkedin.com/in/agostinaperezlopez/

## Licencia

Este proyecto está bajo la licencia [MIT](LICENSE) — ver el archivo `LICENSE` para más detalle.

El dataset CHB-MIT tiene su propia licencia de uso (PhysioNet); no se redistribuye en este
repositorio.

---

## Referencias

- Goldberger, A. et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 2000.
- Shoeb, A. *Application of Machine Learning to Epileptic Seizure Onset Detection and
  Treatment*. PhD thesis, MIT, 2009.
- Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.
- Wang, X. et al. (2025). EEG-based seizure onset detection of frontal and temporal lobe
  epilepsies using 1DCNN. *IEEE TNSRE*.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.
  Springer.
