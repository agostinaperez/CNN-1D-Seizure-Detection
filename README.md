<div align="center">

# Detección Automática de Eventos Epilépticos en Señales EEG mediante CNN-1D

### Clasificación automática de ventanas de EEG para distinguir crisis epilépticas de actividad basal, como apoyo al monitoreo clínico continuo

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-en%20desarrollo-orange.svg)]()

[Demo en vivo](https://TU-USUARIO.github.io/TU-REPOSITORIO/) · [Reportar un bug](../../issues) · [Pedir una funcionalidad](../../issues)

</div>

---

## 📋 Tabla de contenidos

- [Demo](#-demo)
- [Sobre el proyecto](#-sobre-el-proyecto)
  - [Contexto académico](#contexto-académico)
  - [Construido con](#construido-con)
- [El dataset](#-el-dataset)
- [Metodología](#-metodología)
- [Resultados](#-resultados)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Roadmap](#-roadmap)
- [Autores](#-autores)
- [Licencia](#-licencia)
- [Agradecimientos y referencias](#-agradecimientos-y-referencias)

---

## 🎬 Demo

Este proyecto se demuestra de dos maneras:

1. **Página de presentación** (GitHub Pages, carpeta `docs/`): el pitch del proyecto,
   la descripción del enfoque y los resultados principales, sin necesidad de correr código.
   Disponible en `https://TU-USUARIO.github.io/TU-REPOSITORIO/`.

2. **Demo interactiva local** (API + interfaz web): levanta un servidor local que te deja
   elegir cualquiera de tus checkpoints `.pt`, subir un archivo `.edf`, correr la inferencia
   y ver los gráficos (curva de probabilidad, señal EEG y eventos detectados), además de las
   curvas de entrenamiento y la info de cómputo de cada modelo. Ver [Uso](#-uso).

Para levantarla:

```bash
python -m src.api          # API + interfaz en http://127.0.0.1:8000
```

> Nota: las capturas y el GIF de la demo se agregarán en `docs/img/` cuando estén generados.

---

## 📖 Sobre el proyecto

Inspeccionar visualmente un registro de electroencefalograma (EEG) es una tarea lenta y
demandante para los neurólogos: en monitoreos continuos se acumulan horas de señal por
paciente, y la revisión manual no escala ni está exenta de fatiga. Este proyecto entrena un
clasificador automático que distingue, ventana a ventana, los períodos de crisis (actividad
ictal) de la actividad basal (normal).

El enfoque elegido es una **red neuronal convolucional 1D (CNN-1D)** que opera directamente
sobre la señal cruda filtrada, sin necesidad de transformaciones manuales a otros dominios.
El sistema se entrena con aprendizaje supervisado y evalúa su desempeño con métricas
clínicas de detección de crisis (sensibilidad, falsas alarmas por hora y latencia de
detección), no solo con accuracy.

El resultado principal: la CNN-1D alcanza una sensibilidad de crisis del **67,6 %** (23 de 34
crisis detectadas) sobre pacientes no vistos, con un AUPRC de **0,39**, superando de forma
consistente al baseline (un clasificador que siempre predice "no crisis"). El modelo es un
primer MVP de apoyo a la decisión, todavía por debajo de los objetivos clínicos de
sensibilidad y de tasa de falsas alarmas.

### Contexto académico

- **Asignatura:** Inteligencia Artificial
- **Institución:** [Nombre de la institución]
- **Cohorte / año:** 2026
- **Docente:** [Nombre del/la docente]

### Construido con

- [Python 3.11+](https://www.python.org/) (desarrollado y probado en 3.14)
- [PyTorch](https://pytorch.org/) / [torchinfo](https://github.com/TylerYep/torchinfo) — red neuronal y resúmenes de arquitectura
- [MNE](https://mne.tools/) / [SciPy](https://scipy.org/) — carga de EDF, filtrado y procesamiento de señal
- [NumPy](https://numpy.org/) / [scikit-learn](https://scikit-learn.org/) — datos y métricas
- [matplotlib](https://matplotlib.org/) — gráficos de resultados
- [FastAPI](https://fastapi.tiangolo.com/) / [Vue 3](https://vuejs.org/) / [ECharts](https://echarts.apache.org/) — demo interactiva de inferencia
- [uv](https://docs.astral.sh/uv/) — gestión de dependencias y entorno

---

## 📊 El dataset

Se usa el **CHB-MIT Scalp EEG Database**, un dataset clínico de dominio público del Hospital
Infantil de Boston (CHB) en colaboración con el MIT. Son señales temporales continuas, muy
no estacionarias y afectadas por distintos tipos de artefactos (musculares, oculares, de
movimiento de electrodos, interferencia de red).

> ⚠️ El dataset crudo pesa **~64 GB** y **no se incluye en este repositorio**. Se descarga de
> PhysioNet y se apunta con `--data-dir` (o la variable de entorno `CHBMIT_DIR`). Los datos
> ya procesados (splits, estadísticas del escalador) sí están versionados en `data/processed/`.

| Campo | Detalle |
|---|---|
| Fuente | [CHB-MIT Scalp EEG Database (PhysioNet)](https://physionet.org/content/chbmit/1.0.0/) |
| Registros | 24 pacientes (chb01–chb24), EEG continuo a 256 Hz |
| Variables | 16 canales bipolares (montaje TUEV) + anotaciones de crisis (`chbXX-summary.txt`) |
| Variable objetivo | Crisis (ictal) vs. no crisis (basal), a nivel de ventana de 5,12 s |
| Período / cobertura | Monitoreo continuo pediátrico, múltiples sesiones por paciente |
| Licencia del dataset | Público (PhysioNet); uso académico. No redistribuido aquí |

---

## 🔬 Metodología

El pipeline completo es: **carga del EDF → filtrado → segmentación en ventanas → escalado →
entrenamiento de la CNN → selección del punto de operación → evaluación inter-paciente**.
Todo está implementado como módulos ejecutables (`python -m src.*`) en `src/`.

### Análisis exploratorio

Las señales EEG son continuas, no estacionarias y con fuerte desbalance entre clases: las
crisis representan una fracción mínima del tiempo total de registro. Los artefactos y el
ruido contaminan la señal útil, y la duración de las crisis varía mucho entre pacientes
(hay crisis muy cortas, de ~8 s, que son las más difíciles). Este desbalance y la
variabilidad inter-paciente motivan las decisiones de diseño siguientes.

### Preprocesamiento

- **Filtrado pasabanda** 0,5–50 Hz con `sosfiltfilt` (fase cero), orden 5.
- **Segmentación** en ventanas nominales de 5,12 s (1310 muestras a 256 Hz) con 50 % de
  solapamiento, siguiendo el montaje TUEV de 16 canales.
- **Escalado robusto** por canal (mediana e IQR, `RobustScaler`), calculado **solo** sobre
  los pacientes de entrenamiento para evitar fuga de datos.
- **Balanceo** mediante submuestreo de la clase mayoritaria (`NEG_POS_RATIO = 3`) y
  **augmentación** con ruido gaussiano solo en ventanas de crisis.

### Modelo(s)

Se entrenó una **CNN-1D** de 6 bloques (Conv1d + BatchNorm + ReLU + MaxPool) seguidos de
Global Average Pooling y una capa densa. Se compararon dos funciones de pérdida:

| Configuración | Pérdida | AUPRC (val) |
|---|---|---|
| BCE (baseline del entrenamiento) | BCE con `pos_weight` | 0,52 |
| Focal `alpha=0.25` | Focal Loss (Lin et al.) | 0,55 |
| Focal `alpha=0.75` | Focal Loss | 0,58 |

La **Focal Loss** pondera menos los ejemplos fáciles y enfoca el entrenamiento en las crisis
cortas y los falsos positivos, que son los casos difíciles.

**Validación**: split **inter-paciente** fijo (train/val/test, sin mezclar ventanas del mismo
paciente entre particiones) para evitar fuga de datos. El punto de operación (umbral) se elige
en validación maximizando la sensibilidad de crisis sujeta a un techo de falsas alarmas
(mediana ≤ 1/h por paciente), con postprocesado a nivel de evento (≥2 positivas en 3 ventanas
consecutivas, intervalo mínimo de 4 min entre alarmas).

---

## 📈 Resultados

Métricas sobre el **test congelado inter-paciente** (6 pacientes, 34 crisis). El baseline es
el clasificador conservador que predice siempre la clase mayoritaria ("no crisis").

| Métrica | Baseline (mayoritaria) | BCE | Focal α=.25 (final) |
|---|---|---|---|
| AUC-ROC | 0,50 | 0,8915 | **0,8957** |
| AUPRC | ≈ 0,002 | 0,3736 | **0,3881** |
| Sensibilidad de crisis (evento) | 0 % | 64,7 % (22/34) | **67,6 % (23/34)** |
| Falsas alarmas/h (mediana por paciente) | 0 | 2,05 | 2,42 |
| Sensibilidad (ventana) | 0 % | 67,3 % | 63,6 % |
| Especificidad (ventana) | 100 % | 95,7 % | **97,2 %** |
| MCC (ventana) | 0 | 0,139 | **0,164** |
| Latencia media de detección | — | 16,70 s | 15,85 s |

Focal α=.25 es superior a BCE en ranking (AUPRC y AUC) y en la mayoría de las métricas de
ventana: reduce los falsos positivos (~36 % menos de FPR) y mejora MCC y F1.

### Análisis crítico

El modelo es un **MVP funcional** que supera al baseline, pero **no alcanza todavía los
objetivos clínicos**:

- La sensibilidad de crisis (67,6 %) queda lejos del objetivo de ≥90 %.
- Las falsas alarmas (~2,4/h medianas) superan el techo clínico de 1/h.
- Las **crisis cortas** siguen siendo el punto débil: en `chb16` (crisis de ~8 s) solo se
  detectan 2 de 10 para ambos modelos. Es un problema de representación de datos, no de la
  función de pérdida.
- Con 34 crisis de test, la mejora de una crisis entre BCE y Focal tiene aún mucha
  incertidumbre estadística.

Con más tiempo se atacaría el desbalance de crisis cortas, se estimaría la variabilidad
inter-paciente con validación cruzada (ver [`PLAN_CV.md`](PLAN_CV.md)) y se explorarían
arquitecturas híbridas (CNN+LSTM o CNN 2D sobre espectrogramas).

---

## 📂 Estructura del proyecto

```text
.
├── README.md
├── pyproject.toml
├── LICENSE
├── PLAN_CV.md             # plan de validación cruzada (próximo paso)
├── INFORME_TECNICO.pdf    # informe técnico del trabajo
├── src/                   # código fuente (funciones y módulos propios)
│   └── api.py             # API FastAPI + demo de inferencia
├── frontend/              # interfaz web de la demo (Vue 3 + ECharts)
├── notebooks/             # notebooks de exploración y experimentación
├── data/
│   ├── raw/               # CHB-MIT crudo (~64 GB, NO versionado)
│   └── processed/         # split.json, scaler_stats.npz, folds de CV
├── models/                # checkpoints entrenados (.pt, NO versionados)
├── inferences/            # reportes JSON de inferencia
├── results/               # métricas de evaluación (.txt)
├── tests/                 # tests (vacío por ahora)
└── docs/                  # demo estática para GitHub Pages
    ├── index.html
    ├── img/
    └── assets/
```

---

## ⚙️ Instalación

### Prerrequisitos

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git
- El dataset CHB-MIT descargado de [PhysioNet](https://physionet.org/content/chbmit/1.0.0/)
  (~64 GB, no incluido en el repo).

### Pasos

```bash
# 1. Cloná el repositorio
git clone https://github.com/TU-USUARIO/TU-REPOSITORIO.git
cd TU-REPOSITORIO

# 2. Creá el entorno virtual
uv venv --python 3.11

# 3. Activalo
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Instalá las dependencias (el paquete `src` se instala en modo editable)
uv pip install -e .

# 5. Configurá la ruta del dataset
#    Copiá el template y completá CHBMIT_DIR con la ruta local del dataset descargado:
#    Windows (cmd):  copy .env.example .env
#    Linux/Mac:      cp .env.example .env
```

Descargá el dataset de PhysioNet (~64 GB, no incluido en el repo) y setea su ruta en `.env`
(`CHBMIT_DIR`). Alternativamente podés pasar `--data-dir <ruta>` a cada script sin usar `.env`.

---

## ▶️ Uso

El flujo completo de punta a punta:

```bash
# Split inter-paciente (train/val/test) -> data/processed/split.json
python -m src.split --data-dir <ruta/chbmit/1.0.0>

# Stats del escalador robusto -> data/processed/scaler_stats.npz
python -m src.preprocessing --compute-stats --data-dir <ruta/chbmit/1.0.0>

# Entrenar (BCE y Focal)
python -m src.train --data-dir <ruta/chbmit/1.0.0> --out models/bce.pt --loss bce --seed 42
python -m src.train --data-dir <ruta/chbmit/1.0.0> --out models/focal.pt --loss focal --seed 42

# Evaluar el checkpoint elegido sobre test
python -m src.evaluate --checkpoint models/focal.pt --split test --data-dir <ruta/chbmit/1.0.0>

# Inferencia sobre un EDF suelto (no necesita labels)
python -m src.inference --input <archivo.edf> --checkpoint models/focal.pt
```

La inferencia procesa un EDF pregrabado completo y reporta, por ventana, si detecta crisis;
el informe detallado se guarda en `inferences/`.

### Demo interactiva (API + interfaz web)

```bash
# Levanta la API y sirve la interfaz (Vue + ECharts)
python -m src.api
# o, si preferís uvicorn directamente:
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Abrí `http://127.0.0.1:8000` en el navegador. La interfaz permite:

- **Elegir un modelo** de entre los checkpoints `.pt` de `models/` y ver su historial de
  entrenamiento (curvas de loss y métricas), la info de cómputo (parámetros, MACs/FLOPs,
  latencia) y su configuración.
- **Subir un EDF** (con el selector de archivos local), correr la inferencia y ver la curva
  de probabilidad de crisis por ventana, la señal EEG con los eventos detectados resaltados
  y el detalle de cada evento (inicio, fin, duración).

---

## 🗺️ Roadmap

- [ ] **Validación cruzada agrupada por paciente** para estimar la generalización real
      (planificada en [`PLAN_CV.md`](PLAN_CV.md)).
- [ ] Mejorar la detección de **crisis cortas** (más datos, pérdidas específicas o
      arquitecturas híbridas CNN+LSTM).
- [ ] Exportar el modelo a ONNX para inferencia más liviana y portable.

---

## 👥 Autores

- **[Nombre Apellido]** — [GitHub](https://github.com/TU-USUARIO) · [LinkedIn](https://linkedin.com/in/usuario)

---

## 📄 Licencia

Este proyecto está bajo la licencia [MIT](LICENSE) — ver el archivo `LICENSE` para más detalle.

El dataset CHB-MIT tiene su propia licencia de uso (PhysioNet); no se redistribuye en este
repositorio.

---

## 🙏 Agradecimientos y referencias

- Goldberger, A. et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 2000.
- Shoeb, A. *Application of Machine Learning to Epileptic Seizure Onset Detection and
  Treatment*. PhD thesis, MIT, 2009.
- Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.
- Wang, X. et al. (2025). EEG-based seizure onset detection of frontal and temporal lobe
  epilepsies using 1DCNN. *IEEE TNSRE*.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*.
  Springer.

La lista completa de referencias del trabajo figura en [`PLAN_CV.md`](PLAN_CV.md) y en el
[`INFORME_TECNICO.pdf`](INFORME_TECNICO.pdf).
