
#config.py — Configuración centralizada del proyecto.
#Rutas, hiperparámetros y decisiones de diseño

from __future__ import annotations
import os
from pathlib import Path

# Rutas del proyecto

# Directorio raíz: ia-final/
BASE_DIR = Path(__file__).resolve().parent.parent # .resolve() normaliza la ruta, .parent sube un nivel por vez.
# Carpeta de datos procesados (split.json, ventanas, stats).
PROCESSED_DIR = BASE_DIR / "data" / "processed"
# Carpeta de checkpoints del modelo entrenado.
MODELS_DIR = BASE_DIR / "models"
# Ubicación de las anotaciones procesadas del split.
SPLIT_FILE = PROCESSED_DIR / "split.json"

# Dataset CHB-MIT: se setea por argumento en cada script; este es el default local. En Google Colab se pasa con --data-dir la ruta de Google Drive.
DATASET_DIR = Path(os.environ.get("CHBMIT_DIR", r"C:/Users/Usuario/tesis/physionet.org/files/chbmit/1.0.0",))

# Stats del estandarizador por canal
SCALER_STATS_FILE = PROCESSED_DIR / "scaler_stats.npz"

# Señal y segmentación
# Frecuencia de muestreo en Hz
FS = 256
# según nyquist con 256 muestras puedo representar máx 128 Hz, pero mi pasabanda llega a 50Hz así q no tengo riesgo de aliasing
# Duración nominal de cada ventana de análisis (segundos).
WIN_SECONDS = 5.12
# Muestras por ventana: redondeamos HACIA ABAJO -> 1310 (par).
# La duración efectiva es WIN_SAMPLES / FS = 5.1171875 s.
WIN_SAMPLES = int(FS * WIN_SECONDS)
# Duración física real de las ventanas después del redondeo a muestras.
WIN_SECONDS_EFFECTIVE = WIN_SAMPLES / FS

# Muestras por ventana finales (usadas por todo el pipeline).
WIN_SAMPLES_USED = WIN_SAMPLES
# Solapamiento entre ventanas contiguas (50%).
OVERLAP = 0.5
# Paso (stride) entre inicios de ventana: 50% de la ventana = 655 muestras.
STRIDE_SAMPLES = int(WIN_SAMPLES_USED * (1.0 - OVERLAP))
# Filtrado pasabanda de frecuencias baja (0.5 Hz, deriva lenta) y alta (50 Hz, ruido de red).
LOW_FREQ = 0.5
HIGH_FREQ = 50.0

# Orden del filtro Butterworth (5 es un buen compromiso entre pendiente y estabilidad numérica).
# igual,5 no es el orden efectivo que realmente se aplica. sosfiltfilt filtra dos veces (adelante y atrás), y
# por lo tanto CUADRA la respuesta en magnitud: el orden efectivo del filtro real es 10, atenuando el doble (mejor)
FILTER_ORDER = 5 #q tan "bruscamente" se corta la señal

# Escalado robusto por canal (RobustScaler de scikit-learn).
#  x' = (x - mediana_c) / IQR_c   con IQR = Q75 - Q25 (rango intercuartílico).
SCALER = "robust"

# Rango de cuantiles del RobustScaler (defaults de scikit-learn)
ROBUST_QUANTILE_RANGE = (25.0, 75.0)

# Submuestreo para estimar mediana/IQR sobre TODAS las ventanas de train sin guardarlas
STATS_WINDOW_STRIDE = 16
STATS_SAMPLE_STRIDE = 16


# Canales compatibles con el dataset TUEV por si se implementa transfer learning
CHANNELS_TUEV = ["FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3","C3-P3","P3-O1","FP2-F4","F4-C4","C4-P4","P4-O2","FP2-F8","F8-T8","T8-P8","P8-O2"]

N_CHANNELS = len(CHANNELS_TUEV)

# Semilla global para reproducibilidad (datos, batches e inicialización).
SEED = 42

# Split por paciente (inter-paciente). TEST_RATIO bajo (0.21) apunta a ~6 pacientes
# de test para liberar más pacientes al pool de desarrollo (train + val).
TEST_RATIO = 0.21
N_VAL_PATIENTS = 5 # Cantidad de pacientes que se reservan de train para VALIDACIÓN.
# este paciente tiene q quedar en validación (val representativo de casos difíciles), este tiene crisis cortas que me hacen renegar así q las tiene q tener en cuenta para calibrar el umbral.
VAL_REQUIRED_PATIENTS = ["chb14"]

# Peso relativo de los SEGUNDOS DE CRISIS frente a los ARCHIVOS en el reparto
# codicioso del split. Los segundos de crisis determinan cuántas ventanas
# POSITIVAS caen en test (de eso depende la sensibilidad medida); los archivos
# (~horas) determinan el volumen total (de eso depende el FPR).
# Valor 2.0 = la fracción de crisis pesa el doble que la de archivos: deja
# ambas fracciones de test cerca de 0.30 (0.306 archivos / 0.186 crisis).
# Valor 1.0 = test con muchas horas pero pocas crisis (0.314/0.129).
SPLIT_W_ZSEC = 2.0

# Hiperparámetros de entrenamiento:-----------------------------------------------------------------------

#learning rate inicial. Yo uso optimizador AdamW por ende el learning rate es de tasa adaptativa, el de cada peso se va recalculando y ajustando
LEARNING_RATE = 1e-4
# Clip de gradiente (saco la norma L2, sumando cada gradiente de la lista elevado al cuadrado, y sacando la raiz cuadrada del total): si la norma total es mayor a grad_clip, saco el factor
#de escala -> factor = grad_clip / norma L2. A ese factor de escala lo multiplico x cada elemento de la lista de gradientes, para achicarlos proporcionalmente.
#esto hace q no haga saltos como loco el gradiente y no se me desbalancee y me oscile tanto el modelo
GRAD_CLIP = 1.0
# learning rate mínimo del scheduler coseno (se alcanza al final del entrenamiento).
LR_MIN = 0.0
# Weight decay del AdamW: penaliza pesos grandes (regularización L2). En AdamW se aplica "desacoplado" del momento adaptativo (a diferencia del Adam clásico)
# 1e-2 es el default de PyTorch; se puede bajar a 1e-4 si se nota underfitting.
WEIGHT_DECAY = 1e-2
# Batch: cantidad de ventanas que ve la red antes de cada paso de gradiente.
BATCH_SIZE = 64
# Máximo de épocas (una época = 1 pasada por todas las ventanas muestreadas).
EPOCHS = 40
# Early stopping: cortar si la pérdida de validación no mejora en 8 épocas.
PATIENCE = 8

# Ratios de negative:positive en cada batch de train.Ej: 4 => 4 ventanas no-crisis por cada ventana de crisis.
NEG_POS_RATIO = 3
# Dropout para regularización del modelo.
DROPOUT = 0.4 #en cada lote apaga al 40% de las neuronas al azar, así no se sobreajusta

# Augmentación con ruido gaussiano (paper 1DCNN IEEE TNSRE 2025): se suma ruido
# N(0, NOISE_STD) SOLO a las ventanas de crisis (clase minoritaria) durante el train.
# El desvío está en unidades de la señal ya escalada (RobustScaler: mediana~0, IQR~1).
# NOISE_STD=0 desactiva el ruido.
NOISE_STD = 0.1

# Umbral de decisión de la clasificación binaria:
#la red no devuelve un sí o un no, sino un valor entre 0 y 1, q es la probabilidad de q haya crisis. Si la probabilidad es mayor al
#treshold, se considera q la red predijo crisis. Es ajustable para priorizar sensibilidad o especificidad
THRESHOLD = 0.75

# Criterio de selección del punto de operación a nivel EVENTO: techo clínico de
# falsas alarmas por hora medido como MEDIANA por paciente (robusto a un paciente ruidoso). Entre los umbrales con mediana <= MAX_FALSE_ALARMS_PER_HOUR se elige el
# de MAYOR sensibilidad de crisis (desempate: menor mediana de falsas alarmas).
MAX_FALSE_ALARMS_PER_HOUR = 5.0

# Postprocesado a nivel EVENTO (crisis), como en el paper 1DCNN IEEE TNSRE 2025:
# dispara una alarma si hay >= POSITIVES_FOR_EVENT predicciones positivas dentro de las últimas WINDOW_RANGE_FOR_EVENT ventanas consecutivas.
# 2-de-3: regla más sensible que 3-de-4, para no perder crisis cortas.
POSITIVES_FOR_EVENT = 2
WINDOW_RANGE_FOR_EVENT = 3
# Intervalo mínimo entre dos alarmas, para no contar la misma crisis dos veces.
# 4 min (240 s): más conservador que 30 s, alinea el punto de operación con el
# estándar clínico de la literatura (Wang TNSRE 2025 usa 20 min).
EVENT_MIN_ALARM_INTERVAL = 240.0
# Una crisis cuenta como DETECTADA si hay una alarma dentro de [onset, onset + MAX_LATENCY].
EVENT_MAX_LATENCY = 30.0
#--------------------------------------------------------------------------------------------------------------------
# Workers del DataLoader (0 = proceso único)
NUM_WORKERS = 0

# Arquitectura CNN-1D. se usa por model.py
# Filtros (canales) por cada bloque Conv1D.
CONV_CHANNELS = [64, 128, 256, 256, 256, 256]
# Kernel de cada bloque Conv1D.
CONV_KERNELS = [7, 5, 5, 3, 3, 3]
# Neurons de las capas densas finales.
FC_UNITS = 256
