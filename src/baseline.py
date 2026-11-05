"""
Baselines clásicos sobre potencia por bandas de frecuencia.

Este módulo implementa un baseline de modelos de machine learning clásicos (regresión logística y SVM lineal) entrenados sobre features de ingeniería. Se entrenan dos modelos desde cero sobre las features espectrales. La CNN no se re-entrena, solo se compara contra su resultado.
La CNN consume la señal cruda (16 canales x 1310 muestras) y aprende sola la representación, mientras que los baselines necesitan features calculadas a mano. La hipótesis
clínica es que la crisis epiléptica tiene una "firma" espectral: spikes rápidos (beta/gamma), enlentecimiento (theta/delta) y cambios de balance en alpha.

Según Fourier, puedo expresar cualquier señal como suma de senos y cosenos, c/u con una frecuencia y amplitud distinta. En el EEG, partes distintas del cerebro generan ondas que oscilan a ritmos distintos
por lo q separar estas frecuencias sirve para clasificar la act cerebral. La potencia es la amplitud al cuadrado, promediada, y es una medida de cuánta energía aporta esa onda a la señal total.

Para cada ventana de 5,12 s se calcula:
  1. Densidad espectral de potencia (la distribucion de la potencia a lo largo de las frecuencias, me dice en dónde se concentra la energía. Calculo su estimación con el método de welch, en el que tomo las 1310 muestras correspondientes a una ventana, la divido en segmentos de 256 muestras con 50% de solapamiento, calculo el cuadrado de la magnitud de la FFT de cada segmento y los promedio (para q no sea tan ruidoso))
  2. Potencia integrada en 5 bandas canónicas (absolutas): delta, theta, alpha, beta, gamma. (Sería la integral de la densidad espectral en cada banda, que es equivalente a sumar la potencia de todas las frecuencias que caen dentro de esa banda). Esto da 16 canales x 5 bandas = 80 features absolutas.
  3. 80 potencias relativas (banda / total): Las mismas 80 potencias absolutas de antes pero cada una dividida por la potencia total de la ventana, invariantes a la amplitud global.
  4. 32 ratios (theta/alpha y alpha/beta por canal): balance espectral entre bandas.  θ/α: relación entre enlentecimiento (theta) y el ritmo de reposo (alpha). Cuando el cerebro "se enlentece", este ratio sube. α/β: relación entre actividad de reposo y actividad rápida.
  5. 1 potencia total de toda la ventana.
Total: 193 features.

Estos clasificadores de baseline entonces aprenden una regla sobre esos 193 features. Se encuentra q combinación de estos números separa mejor "crisis" de "no crisis". La limitación es que no toma en cuenta la morfología x ejemplo,
o la estructura temporal dentro de la ventana, a diferencia de la CNN q aprende patrones de forma y tiempo también y no sobre features sino sobre la señal cruda

Se reutiliza del pipeline de la CNN:
  - El mismo split inter-paciente (train/val/test de split.json).
  - Las mismas ventanas y etiquetas.
  - El mismo balanceo de train (todas las positivas + 3x negativas, NEG_POS_RATIO).
  - El mismo postprocesado de evento (2-de-3 ventanas + intervalo de 4 min).
  - La misma selección de punto de operación (máx sensibilidad de evento con <=1 FA/h
    en validación) y el mismo test congelado.
  - Escalado de features ajustado SOLO en train (sin fuga de datos), igual que el
    RobustScaler de la CNN se ajusta solo con train.

    python -m src.baseline --data-dir <ruta/chbmit/1.0.0> --model logistic
    python -m src.baseline --data-dir <ruta/chbmit/1.0.0> --model svm
    python -m src.baseline --data-dir <ruta/chbmit/1.0.0> --model all

Requiere el dataset CHB-MIT
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.signal
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import (
    DATASET_DIR,
    EVENT_MAX_LATENCY,
    EVENT_MIN_ALARM_INTERVAL,
    FS,
    MAX_FALSE_ALARMS_PER_HOUR,
    NEG_POS_RATIO,
    POSITIVES_FOR_EVENT,
    SEED,
    SPLIT_FILE,
    WINDOW_RANGE_FOR_EVENT,
)
from src.data import build_splits_dataloaders
from src.metrics import (
    binary_metrics,
    event_metrics,
    rank_metrics,
    select_event_operating_point,
)
from src.protocol import THRESHOLD_GRID

# Bandas canónicas del EEG en Hz. Son rangos estándar de la neurofisiología: Kane N et al. A revised glossary of terms most commonly used by clinical electroencephalographers and updated proposal for the report format of the EEG findings. Revision 2017. Clin Neurophysiol Pract. 2017 Aug 4;2:170-185. doi: 10.1016/j.cnp.2017.07.002.
#   delta  (0.5-4)  : sueño profundo, patología, enlentecimiento focal.
#   theta  (4-8)    : somnolencia, enlentecimiento; en crisis ictales suele aumentar.
#   alpha  (8-13)   : reposo con ojos cerrados (ritmo posterior dominante).
#   beta   (13-30)  : actividad rápida, alerta, artefacto muscular, spikes.
#   gamma  (30-50)  : actividad rápida de alta frecuencia (asociada a crisis).
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}
N_BANDS = len(BANDS)
N_CHANNELS_FEAT = 16
# 16 canales x 5 bandas (absolutas) + 16 x 5 (relativas) + 16 x 2 (ratios) + 1 (total)
N_FEATURES = N_CHANNELS_FEAT * N_BANDS * 2 + N_CHANNELS_FEAT * 2 + 1

# Valor ínfimo para evitar divisiones por cero (potencias pueden ser ~0 en bandas vacías).
_EPS = 1e-12


def band_power_features(windows: np.ndarray) -> np.ndarray:
    """Calcula las features espectrales de un bloque de ventanas.
    Args:
        windows: (n, 16, 1310) señal filtrada (sin escalado robusto).

    Returns:
        (n, 193) matriz de features float32.
    """
    windows = np.asarray(windows)
    if windows.ndim != 3 or windows.shape[0] == 0:
        return np.empty((0, N_FEATURES), dtype=np.float32)

    # welch opera sobre la última dimensión (el tiempo), y calcula el espectro x separado para cada combinacion de ventana y canal. No mezcla.
    # freqs: el vector de las frecuencias analizadas (0 .. Nyquist)
    # psd: potencia estimada por ventana, por canal, por frecuencia
    freqs, psd = scipy.signal.welch(windows, fs=FS, nperseg=256, axis=-1)  # psd: (n, 16, F), o sea reemplazo el eje de tiempo x el de frecuencia

    # Máscara booleana por banda: qué frecuencias caen dentro de [lo, hi).
    masks = {name: (freqs >= lo) & (freqs < hi) for name, (lo, hi) in BANDS.items()}

    # Para cada banda, sumar la potencia de todas sus frecuencias -> potencia total de esa banda. Apilar las 5 bandas -> tensor (n, 16, 5).
    band_powers = np.stack([psd[:, :, masks[name]].sum(axis=-1) for name in BANDS], axis=-1)  # (n, 16, 5)

    # Potencia total de la ventana (suma d todos los canales y bandas).
    total = band_powers.sum(axis=(1, 2), keepdims=True)  # (n, 1, 1)

    # Potencia relativa (banda / total): hace la feature invariante a la amplitud global  del registro para q un paciente +ruidoso no domine
    relative = band_powers / np.maximum(total, _EPS)  # (n, 16, 5)

    # Ratios deespectral balance
    theta = band_powers[:, :, 1]
    alpha = band_powers[:, :, 2]
    beta = band_powers[:, :, 3]
    ratio_theta_alpha = theta / np.maximum(alpha, _EPS)  # (n, 16)
    ratio_alpha_beta = alpha / np.maximum(beta, _EPS)  # (n, 16)

    n = windows.shape[0]
    # reshape(n, -1) aplana cada tensor a una fila por ventana; concatenate(axis=1) las junta lado a lado así m queda una única matriz (n, 193)
    features = np.concatenate(
        [
            band_powers.reshape(n, -1),        # 80
            relative.reshape(n, -1),           # 80
            ratio_theta_alpha.reshape(n, -1),  # 16
            ratio_alpha_beta.reshape(n, -1),   # 16
            total.reshape(n, 1),               # 1
        ],
        axis=1,
    )
    return features.astype(np.float32)


def extract_dataset_features(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Extrae features de TODAS las ventanas de un WindowDataset, en orden de archivo. lo uso en val y test xq necesito la distribución natural completa, no
    balanceada, para calcular métricas comparables a las de la CNN. El orden en que se concatenan las features coincide con dataset.file_ids / dataset.local_ids, así que
    las métricas a nivel evento se pueden calcular bien
    """
    dataset._ensure_index()  # arma el índice global ventana -> archivo (cabeceras)
    feat_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    for path in dataset.valid_files:
        res = dataset._get_processed(path)  # filtra, segmenta y devuelve windows + labels
        feat_blocks.append(band_power_features(res["windows"]))
        label_blocks.append(res["labels"])
    features = np.concatenate(feat_blocks) if feat_blocks else np.empty((0, N_FEATURES), dtype=np.float32)
    labels = np.concatenate(label_blocks) if label_blocks else np.empty(0, dtype=np.int8)
    return features, labels


def extract_train_features(dataset, neg_pos_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Features de train procesando solo los archivos necesarios para el balanceo.
    Se usa el índice global (armado solo con cabeceras, sin cargar señal) para saber qué archivos tienen ventanas de crisis, cuántas negativas hay por archivo. Incluye todos los archivos con alguna ventana
    positiva y, si hacen falta más negativas, agrega archivos que solo tienen negativas hasta llegar al ratio con el q voy a trabajar (el mismo q en CNN). Me ahorro procesar todos los archivos al vicio digamos
    """
    dataset._ensure_index()
    labels = dataset.labels
    file_ids = dataset.file_ids
    n_files = len(dataset.valid_files)

    # bincount cuenta cuántas ventanas positivas/negativas hay POR archivo
    pos_per_file = np.bincount(file_ids[labels == 1], minlength=n_files)
    neg_per_file = np.bincount(file_ids[labels == 0], minlength=n_files)

    # Archivos con al menos una crisis.
    pos_files = list(np.flatnonzero(pos_per_file > 0))
    n_pos = int(pos_per_file.sum())
    target_neg = int(n_pos * neg_pos_ratio)  # cuántas negativas quiero en total (3x)
    covered_neg = int(neg_per_file[pos_files].sum())

    # Archivos negativos, al azar para no sesgar
    rng = np.random.default_rng(seed)
    neg_only = np.flatnonzero((neg_per_file > 0) & (pos_per_file == 0))
    rng.shuffle(neg_only)
    extra_neg_files: list[int] = []
    remaining = target_neg - covered_neg
    for f in neg_only:
        if remaining <= 0:
            break
        extra_neg_files.append(int(f))
        remaining -= int(neg_per_file[f])

    files_to_process = [dataset.valid_files[f] for f in pos_files + extra_neg_files]

    feat_blocks: list[np.ndarray] = [] #las caracteristicas para cada archivo
    label_blocks: list[np.ndarray] = []#las etiquetas q correspondan a c archivo
    for path in files_to_process:
        res = dataset._get_processed(path)
        #esta funcioon devuelve una matriz de caracteristicas de forma (n_ventanas, 193)
        feat_blocks.append(band_power_features(res["windows"])) #lo llamo 1 vez x archivo y le paso todas sus ventanas
        label_blocks.append(res["labels"])
    features = np.concatenate(feat_blocks) if feat_blocks else np.empty((0, N_FEATURES), dtype=np.float32)
    labels = np.concatenate(label_blocks) if label_blocks else np.empty(0, dtype=np.int8)
    return features, labels


def _subsample_train(features: np.ndarray, labels: np.ndarray, neg_pos_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Todas las positivas + (neg_pos_ratio x positivas) de negativas al azar.
    Mismo balanceo que hace la CNN en train, así la comparación es justa.
    """
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    if pos.size == 0:
        return features, labels
    rng = np.random.default_rng(seed)
    n_neg = min(int(pos.size * neg_pos_ratio), neg.size)
    chosen = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
    return features[chosen], labels[chosen]


def make_classifier(name: str):
    """Construye el clasificador. Cada ventana ya se transformó en una fila con 193 features, y la etiqueta correspondiente a esa ventana me dice si hay o no crisis ahí.

    Regresión logística: modelo lineal para clasificación binaria. Aprende pesos `w` y sesgo `b`, y predice p(crisis) = sigmoid(w·x + b). Es interpretable (cada feature
    tiene un peso) y devuelve directamente una probabilidad calibrada en [0, 1]. `class_weight="balanced"` le da más peso a la clase minoritaria (crisis), compensando el desbalance extremo.

    SVM lineal:  busca el hiperplano (frontera que el modelo aprende para separar las dos clases) que separa las clases con el máximo margen (la mayor
    distancia a los ejemplos más cercanos de cada clase). Uso LinearSVC porque funciona mejor para conjuntos grandes, las otras alternativas pueden aprender fronteras
    más curvas pero el costo de entrenamiento crece mucho. Además, este linear solo expone decision_function q es un score, no una probabilidad
    para poder comparar con el mismo umbral [0,1] de la CNN se calibra con Platt scaling (CalibratedClassifierCV con method="sigmoid"), que ajusta una sigmoide score->prob con validación cruzada interna (cv=3).
    """
    if name == "logistic":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "svm":
        # LinearSVC no expone predict_proba; se calibra con sigmoide (Platt) para operar en [0, 1].
        return CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=5000),
            method="sigmoid",
            cv=3,
        )
    raise ValueError(f"Modelo no soportado: {name!r}")


def _report_probs(model, scaler, X):
    """Escala las features y devuelve la probabilidad de la clase positiva (crisis).

    predict_proba devuelve (n, 2) con [p(no crisis), p(crisis)]; se toma la columna 1.
    """
    return model.predict_proba(scaler.transform(X))[:, 1]


def _print_model_report(name: str, probs, labels, *, file_ids, local_ids, valid_files, annotations, threshold):
    """Reporta las MISMAS métricas que usa la CNN para un modelo clásico.

      - binary_metrics: sensibilidad, especificidad, FPR y falsos positivos/hora a nivel
        de ventana (matriz de confusión).
      - rank_metrics: AUC-ROC y AUPRC, independientes del umbral (calidad de ranking).
      - event_metrics: sensibilidad de crisis, falsas alarmas de evento/hora y latencia a
        nivel evento, con el postprocesado 2-de-3 y el intervalo mínimo de 4 min.
    """
    m = binary_metrics(probs, labels, threshold, file_ids=file_ids, local_ids=local_ids)
    rk = rank_metrics(probs, labels)
    ev = event_metrics(
        probs, labels,
        file_ids=file_ids, local_ids=local_ids,
        valid_files=valid_files, annotations=annotations,
        threshold=threshold,
        n_within=POSITIVES_FOR_EVENT, n_window=WINDOW_RANGE_FOR_EVENT,
        min_alarm_interval=EVENT_MIN_ALARM_INTERVAL, max_latency=EVENT_MAX_LATENCY,
    )
    auc = f"{rk['auc_roc']:.4f}" if rk["auc_roc"] is not None else "—"
    auprc = f"{rk['auprc']:.4f}" if rk["auprc"] is not None else "—"
    print(f"  {name}")
    print(f"    [ranking]   AUC-ROC={auc}  AUPRC={auprc}")
    print(f"    [ventana]   sens={m['sensibility']:.4f}  spec={m['specificity']:.4f}  "
          f"fpr={m['false_positive_rate']:.4f}  fp/h={m['false_positive_per_hour']:.2f}")
    print(f"    [evento]    sens={ev['sensibility']:.4f} ({ev['n_detected']}/{ev['n_seizures']})  "
          f"fa/h={ev['event_false_alarms_per_hour']:.2f} (mediana {ev['median_false_alarms_per_hour']:.2f})  "
          f"latencia={ev['latency_mean']:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline clásico (potencia por banda) vs. CNN-1D.")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--split-file", type=str, default=str(SPLIT_FILE))
    parser.add_argument("--model", type=str, default="all", choices=["logistic", "svm", "all"])
    parser.add_argument("--neg-pos-ratio", type=float, default=NEG_POS_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit-files", type=int, default=None, help="Limitar N EDFs por split (smoke-test).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir no es válido: {data_dir}")

    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    models = ["logistic", "svm"] if args.model == "all" else [args.model]

    print(f"Baseline clásico: {', '.join(models)}  (features={N_FEATURES})")
    print(f"  data-dir: {data_dir}  |  neg_pos_ratio={args.neg_pos_ratio}  |  seed={args.seed}")

    # Datasets sin escalado robusto (scaler_stats=None): las features se calculan sobre la señal filtrada CRUDA, porque la potencia de banda quiere amplitudes naturales. El
    # escalado se hace después, a nivel de features, con el StandardScaler (ajustado en train).
    loaders = build_splits_dataloaders(data_dir, split, scaler_stats=None, batch_size=64, num_workers=0, limit_files=args.limit_files)
    t0 = time.time()
    print("\nExtrayendo features por banda...")
    train_ds = loaders["train"]["dataset"]
    val_ds = loaders["val"]["dataset"]
    test_ds = loaders["test"]["dataset"]
    
    X_train, y_train = extract_train_features(train_ds, args.neg_pos_ratio, args.seed)
    X_val, y_val = extract_dataset_features(val_ds)
    X_test, y_test = extract_dataset_features(test_ds)
    print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}  "
          f"({time.time() - t0:.1f}s)")

    for name in models:
        # Balanceo exacto + scaler ajustado solo en train
        Xtr, ytr = _subsample_train(X_train, y_train, args.neg_pos_ratio, args.seed)
        scaler = StandardScaler().fit(Xtr)
        model = make_classifier(name)
        print(f"\nEntrenando {name} sobre {Xtr.shape[0]} ventanas submuestreadas...")
        t1 = time.time()
        model.fit(scaler.transform(Xtr), ytr)
        print(f"  (entrenado en {time.time() - t1:.1f}s)")

        # Punto de operación en validación: misma regla clínica que la CNN (máx sensibilidad de evento con mediana de falsas alarmas <= 1/h). select_event_operating_point barre
        # la grilla de umbrales y devuelve (métricas, umbral) o (None, None) si ninguno cumple.
        val_probs = _report_probs(model, scaler, X_val)
        op, op_threshold = select_event_operating_point(
            val_probs, y_val,
            file_ids=val_ds.file_ids, local_ids=val_ds.local_ids,
            valid_files=val_ds.valid_files, annotations=val_ds.annotations,
            max_false_alarms_per_hour=MAX_FALSE_ALARMS_PER_HOUR,
            n_within=POSITIVES_FOR_EVENT, n_window=WINDOW_RANGE_FOR_EVENT,
            min_alarm_interval=EVENT_MIN_ALARM_INTERVAL, max_latency=EVENT_MAX_LATENCY,
            thresholds=THRESHOLD_GRID,
        )
        if op is None:
            # Fallback explícito: el modelo no logra el techo clínico en ningún umbral.
            print(f"  [WARN] {name}: ningún umbral respetó el techo de falsas alarmas en validación.")
            op_threshold = 0.5

        print(f"  [validación] umbral seleccionado: {op_threshold if op_threshold is not None else 'N/A'}")
        test_probs = _report_probs(model, scaler, X_test)
        _print_model_report(
            name, test_probs, y_test,
            file_ids=test_ds.file_ids, local_ids=test_ds.local_ids,
            valid_files=test_ds.valid_files, annotations=test_ds.annotations,
            threshold=op_threshold,
        )

    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
