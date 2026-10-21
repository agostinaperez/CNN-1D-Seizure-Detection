# Plan de Validación Cruzada (CV) para la CNN-1D de Detección de Crisis

> Objetivo: reemplazar el split fijo actual por una validación cruzada agrupada
> por paciente, para estimar de forma honesta la generalización inter-paciente de
> la CNN-1D y comparar de manera robusta los escenarios (BCE vs Focal Loss,
> `alpha`/`gamma`). El test final (`chb16` + pacientes congelados) **no se toca**
> durante la CV.

---

## 1. Marco teórico

### 1.1 Qué es la validación cruzada

La validación cruzada (CV) es una técnica de estimación del error de
generalización que divide los datos en `K` particiones (o *folds*) y, para cada
una, entrena el modelo sobre `K−1` particiones y lo evalúa sobre la partición
restante, repitiendo el proceso hasta que todas las particiones hayan sido usadas
como conjunto de evaluación exactamente una vez (Hastie, Tibshirani & Friedman,
2009; Kohavi, 1995).

La idea central es que **ningún dato que se use para evaluar haya participado en
el entrenamiento**. A diferencia de evaluar sobre el propio train (que da un
error optimistamente sesgado) o sobre un único split (que da un estimador con
mucha varianza), la CV:

- usa todos los datos de desarrollo tanto para entrenar como para validar;
- produce una **distribución** de medidas (una por fold), de la que se pueden
  calcular media y desvío estándar (Arlot & Celisse, 2010);
- es la práctica estándar para **seleccionar modelos e hiperparámetros** y para
  **estimar el desempeño esperado** frente a datos nuevos (Stone, 1974).

En el extremo, `K = n` (un paciente por fold) se denomina *leave-one-out*
(LOO); con grupos se llama *leave-one-patient-out* (LOPO).

### 1.2 Por qué agrupada por paciente (GroupKFold / LOPO)

En EEG, dos ventanas del **mismo paciente** están altamente correlacionadas:

- las ventanas solapan al 50% (solapamiento temporal);
- la señal de un sujeto comparte características propias (montaje, impedancia,
  estado clínico, tipo de crisis).

Si se hace un `train_test_split` aleatorio sobre ventanas, ventanas del mismo
paciente caen tanto en train como en test, y el modelo aprende un "atributo" del
paciente en vez de la morfología de la crisis. Esto infla las métricas y no mide
generalización real: es la forma clásica de *data leakage* en EEG.

Wang et al. (2025) distinguen explícitamente una estrategia *patient-specific*
(modelo por paciente) de una *patient-cross* (dejar pacientes fuera del
entrenamiento y evaluar sobre ellos), y reportan que el desempeño *patient-cross*
es menor pero es el que tiene sentido clínico, porque en producción el sistema
recibe pacientes nunca vistos. Akan et al. (2025) remarcan el mismo punto a nivel
general: los modelos de deep learning en neuroimagen/neurofisiología tienden a
sobreajustar características específicas del sujeto, por lo que la unidad de
independencia debe ser el paciente.

Por eso la CV correcta acá es **GroupKFold** (o LOPO), donde el grupo es el
paciente: un paciente completo queda en train o en val de un fold, **nunca
repartido entre ambos**.

Adicionalmente, por la naturaleza temporal de la señal, conviene recordar que un
CV ingenuo sobre series temporales también fuga información si no respeta la
estructura temporal (Bergmeir & Benítez, 2012). En este proyecto la estructura
temporal se respeta implícitamente agrupando por paciente y procesando cada EDF
como una secuencia.

### 1.3 Selección de modelo vs estimación del error (el sesgo del doble uso)

Hay una distinción importante (Varma & Simon, 2006):

- **Selección**: elegir entre configuraciones (BCE vs focal, `alpha`, `gamma`)
  usando la CV.
- **Estimación**: reportar "cuánto va a rendir el modelo final" usando la misma
  CV.

Si se usa la CV **a la vez** para elegir la mejor configuración y para reportar
su desempeño, el error reportado queda optimistamente sesgado (se elige la que
salió bien por azar). La solución honesta es:

1. CV sobre el pool de desarrollo para **comparar/ordenar** configuraciones.
2. Elegir la configuración ganadora (por media + estabilidad, no por un único
   fold).
3. Reentrenar la ganadora sobre **todo** el pool de desarrollo.
4. Evaluar **una sola vez** sobre el test congelado, que es el número que se
   reporta como desempeño final.

Este es exactamente el flujo que usa la literatura de detección de crisis:
entrenar/elegir en desarrollo y reportar el test independiente una vez (Wang et
al., 2025; Cao et al., 2025).

### 1.4 Por qué CV y no el split fijo actual

El split fijo actual da **un único número** por configuración. Con 24 pacientes y
crisis muy desbalanceadas (algunos pacientes concentran muchos eventos, otros
uno solo), ese número es muy sensible a qué pacientes cayeron en test:

- Un paciente difícil como `chb16` puede dominar la sensibilidad final.
- No sabemos si la mejora de Focal sobre BCE (`AUPRC 0.3736 → 0.3881`, sensibilidad
  de evento `64.7% → 67.6%`) es un efecto real o una casualidad del split.

La CV da **media ± desvío** de cada métrica, lo que permite:

- detectar configuraciones inestables (buena media pero mucha varianza);
- concluir si Focal es consistente a través de pacientes;
- detectar sobreajuste comparando el desempeño en folds de val contra el test.

---

## 2. Qué aportaría concretamente a este proyecto

Con los resultados actuales en mano (mismo split, `seed=42`, `neg_pos_ratio=3`,
`noise_std=0.1`, `lr=5e-5`):

| Modelo | AUPRC (val) | Sensibilidad evento (val) | FDR mediano (val) |
|---|---:|---:|---:|
| BCE | 0.5228 | 81.08% | 0.421/h |
| Focal `alpha=.25` | 0.5457 | 83.78% | 0.421/h |
| Focal `alpha=.75` | 0.5819 | 75.68% | 0.201/h |

La mejora de Focal es real en esta partición, pero con **un único split no se
puede afirmar** que sea robusta. La CV ayudaría a:

1. **Confirmar o descartar la hipótesis focal > BCE** con intervalos de
   confianza empíricos (media ± std por fold).
2. **Elegir `alpha` y `gamma`** de forma defendible (en vez de probar valores
   uno a uno y mirar test).
3. **Estimar la variabilidad inter-paciente**, que es la incertidumbre clínica
   real: ¿el modelo funciona con el paciente 1 pero no con el 4?
4. **Detectar si las crisis cortas están subrepresentadas**: si en los folds
   donde `chb06`/`chb14` (crisis cortas) quedan en val la sensibilidad se
   desploma, el problema es de representación de datos, no de loss.
5. **Proteger el test**: hoy cada "prueba" (alpha=.25, alpha=.75, ...) se evalúa
   sobre el mismo test, lo que termina filtrando información del test hacia las
   decisiones de diseño. Con CV, el test se evalúa solo al final.

**Qué NO va a resolver la CV**: no corrige por sí sola un modelo que no detecta
crisis cortas, ni reduce las falsas alarmas. Lo que hace es **medir de forma
honesta** cuán grave es ese problema y qué tan dependiente es del split.

---

## 3. Diseño experimental propuesto

### 3.1 Pools de pacientes

- **Test congelado (intocable)**: los pacientes actuales de test
  (`chb16` y compañía). Se reserva `chb16` por ser el caso difícil de crisis
  cortas (~8 s), como está hoy.
- **Pool de desarrollo (CV)**: todos los demás pacientes (los de train + val del
  split actual). Son ~18 pacientes.

### 3.2 Número de folds

Recomendación: **K = 5** con `GroupKFold` sobre el pool de desarrollo (o
K = 6 si se prefiere un val más chico por fold).

Razonamiento de costos: cada fold entrena un modelo. Con 3 configuraciones × 5
folds = 15 entrenamientos (~3.5 h c/u en Colab según el historial actual). Si el
costo es inaceptable, bajar a **K = 4** y/o reducir a **2 configuraciones**
(BCE baseline + la mejor focal). LOPO (18 folds) no es viable en este proyecto.

### 3.3 Configuraciones a comparar (máximo 3)

Mantener **un único factor variable** por comparación:

| ID | loss | `gamma` | `alpha` | propósito |
|---|---|---:|---:|---|
| `bce` | bce | — | — | baseline de referencia |
| `focal_a025` | focal | 2.0 | 0.25 | configuración del paper (Lin et al.) |
| `focal_a075` | focal | 2.0 | 0.75 | más peso a la clase positiva |

Todo lo demás fijo e idéntico entre configuraciones: arquitectura, ventana
5.12 s / 50%, filtro 0.5–50 Hz, `batch_size`, `lr`, `weight_decay`, `patience`,
`neg_pos_ratio=3`, `noise_std=0.1`, `seed`.

Si después se quiere estudiar `gamma`, se arma **otra** matriz manteniendo
`alpha` fijo (p. ej. `gamma ∈ {1, 2, 4}`).

### 3.4 Regla anti-leakage por fold

El scaler (RobustScaler: mediana/IQR) **se recalcula en cada fold usando
exclusivamente los pacientes de train de ese fold**. Reusar el
`scaler_stats.npz` global (calculado sobre todo el pool) filtraría información de
los pacientes de val hacia el escalado.

Esto es una extensión directa de la regla ya implementada: hoy `compute_stats`
calcula las stats solo sobre `split["train"]`; en CV hay que llamarlo con el
`train` de cada fold.

### 3.5 Métricas por fold

Usar las mismas métricas clínicas actuales, para no romper la comparabilidad:

- **Nivel ranking**: AUPRC (primaria para comparar focal vs BCE), AUC-ROC.
- **Nivel evento**: sensibilidad de crisis, FDR (evento/h global y **mediana por
  paciente**), latencia mediana.
- **Nivel ventana** (contexto): sensibilidad, especificidad, MCC.

El criterio de selección por fold es el mismo que ya se usa en `train.py`
(`select_event_operating_point`): máximo sensibilidad de evento sujeto a
mediana de FDR `≤ 1/h`, desempate por FDR y AUPRC.

### 3.6 Agregación y reporte

Para cada configuración y cada métrica, reportar:

```text
media ± desvío estándar  (y min / max por fold)
```

Ejemplo de tabla esperada:

| Config | AUPRC | AUC-ROC | Sens. evento | FDR mediano |
|---|---|---|---|---|
| bce | 0.52 ± 0.06 | ... | ... | ... |
| focal_a025 | 0.55 ± 0.07 | ... | ... | ... |
| focal_a075 | 0.58 ± 0.09 | ... | ... | ... |

Decisión: elegir la configuración con **mejor media** y **varianza aceptable**
(si dos están empatadas en media, la más estable). No elegir por el mejor fold
individual.

---

## 4. Plan de implementación técnica

### 4.1 Refactor previo (mínimo y sin romper nada)

1. **`compute_stats`** (`src/preprocessing.py`): hoy recibe un `split_file` y lee
   `split["train"]`. Refactorizar para aceptar una **lista de pacientes de train**
   (o un `split` dict) además del archivo, de modo que CV pueda llamarlo por
   fold sin escribir `split.json` temporales. Mantener el CLI actual intacto.
2. **`train.py`**: exponer una función reutilizable del ciclo de entrenamiento
   (hoy todo está dentro de `main()`). Extraer algo como
   `train_on_split(data_dir, split, scaler_stats, config, out_dir) -> metrics`
   para que CV pueda invocarla por fold sin duplicar lógica.
3. **Semilla**: derivar una semilla **determinista por fold** (p. ej.
   `seed_base + fold_id`) para reproducibilidad, o reusar la misma semilla en
   todos los folds (recomendado: misma semilla para aislar el efecto del fold).

### 4.2 Nuevo módulo `src/cv.py`

Responsabilidades:

1. **`build_folds(patients_dev, K, seed, stats)`** → lista de `(train_patients,
   val_patients)`.
   - Usar `sklearn.model_selection.GroupKFold(n_splits=K)` con grupos = paciente.
   - Opcional: ordenar pacientes por severidad antes para que cada fold sea
     representativo en segundos de crisis.
   - Forzar que cada fold de val contenga al menos un paciente con crisis cortas
     (para que la selección del umbral "vea" el caso difícil), de forma análoga a
     `VAL_REQUIRED_PATIENTS`.
2. **`run_cv(config, K, seed, ...)`** → para cada fold:
   - construir el `split` dict `{train, val}`;
   - calcular scaler sobre el train del fold (anti-leakage);
   - entrenar con `train_on_split(...)`;
   - calcular métricas de val (AUPRC, evento) y guardarlas.
3. **`aggregate(results)`** → media, std, min, max por métrica y configuración.
4. **`refit_final(config)`** → reentrenar la ganadora sobre **todo** el pool de
   desarrollo (train + val), guardar checkpoint final.

Salidas en disco (nuevo directorio `results/cv/`):

```text
results/cv/
  folds.json                     <- definición de folds (reproducible)
  <config>_fold<i>.pt            <- checkpoint por fold (opcional)
  <config>_fold<i>.history.json  <- historia por fold
  summary_cv.json                <- métricas agregadas (media ± std)
  best_config.json               <- configuración ganadora
```

### 4.3 Pseudocódigo de alto nivel

```python
dev_patients = sorted(patients)  # excluye los de TEST congelado
folds = build_folds(dev_patients, K=5, seed=42, stats=patient_seizure_stats(...))

results = {}
for config in CONFIGS:
    fold_metrics = []
    for i, (train_pts, val_pts) in enumerate(folds):
        split = {"train": train_pts, "val": val_pts}
        scaler = compute_stats(train_pts)          # scaler SOLO de este fold
        metrics = train_on_split(split, scaler, config, fold=i)
        fold_metrics.append(metrics)               # AUPRC, sens_evento, FDR, ...
    results[config] = aggregate(fold_metrics)      # media ± std

best = select_best_config(results)                 # media + estabilidad
final_ckpt = refit_final(best, dev_patients)       # entrena sobre train+val
evaluate_once(final_ckpt, test_patients)           # test congelado, UNA vez
```

### 4.4 Decisiones de configuración nuevas

Agregar a `config.py`:

```python
CV_FOLDS = 5                 # número de folds de la CV de desarrollo
CV_SEED = 42                 # semilla de la partición de folds
CV_CONFIGS = ["bce", "focal_a025", "focal_a075"]   # a comparar
```

(Se dejan como constantes para que el notebook/CLI las pueda pisar.)

### 4.5 CLI nuevo

```bash
python -m src.cv \
    --data-dir <ruta/chbmit> \
    --folds 5 \
    --seed 42 \
    --out-dir results/cv
```

Opcionalmente `--configs bce,focal_a025,focal_a075` y `--device cuda`.

---

## 5. Riesgos, decisiones y buenas prácticas

1. **Costo computacional**: 15 entrenamientos. Mitigar con K=4, menos
   configuraciones, o `--limit-files` solo para smoke-test (nunca para el
   resultado final).
2. **Sesgo de doble uso**: reportar el desempeño final solo del test congelado;
   la CV se usa para comparar, no como número final (Varma & Simon, 2006).
3. **Stratificación**: GroupKFold no estratifica por severidad por defecto. Si la
   distribución de crisis por fold queda muy despareja, ordenar pacientes por
   severidad antes de plegar, o garantizar un paciente de crisis cortas por fold.
4. **Comparabilidad entre configuraciones**: mismo split de folds, misma semilla,
   mismo escalador por fold, misma arquitectura y mismo criterio de selección.
   Cualquier cambio adicional invalida la comparación.
5. **Reproducibilidad**: guardar `folds.json` con los pacientes de cada fold y el
   `seed`, de modo que cualquier corrida futura pueda reproducir la misma
   partición.
6. **Métricas con varianza alta por desbalance**: la sensibilidad de evento con
   pocas crisis tiene alta varianza; reportarla siempre con el conteo
   (`detectadas/totales`), no solo el porcentaje.

---

## 6. Criterios de aceptación

La CV se considera lista cuando:

- [ ] los folds se generan agrupando por paciente y quedan documentados en
      `folds.json`;
- [ ] el scaler se recalcula por fold usando solo el train del fold;
- [ ] cada fold reporta AUPRC, AUC-ROC, sensibilidad de evento, FDR (global y
      mediana por paciente) y latencia;
- [ ] el resumen entrega `media ± std` por configuración y métrica;
- [ ] el `chb16` y el resto del test **nunca** aparecen en train/val de ningún
      fold;
- [ ] el modelo final se reentrena sobre todo el pool de desarrollo y se evalúa
      una única vez sobre test;
- [ ] hay al menos un test unitario para `build_folds` (sin fugas entre folds) y
      para el scaler por fold.

---

## 7. Referencias (APA)

- Akan, T., Akan, S., Alp, S., Ledbetter, C. R., Tafti, A. P., Arevalo, O., &
  Bhuiyan, M. A. N. (2025). Deep learning in neuroimaging for neurodegenerative
  diseases: State-of-the art, challenges, and opportunities. *Journal of the
  Neurological Sciences, 478*, 123735. https://doi.org/10.1016/j.jns.2025.123735
- Arlot, S., & Celisse, A. (2010). A survey of cross-validation procedures for
  model selection. *Statistics Surveys, 4*, 40–79. https://doi.org/10.1214/09-SS054
- Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time
  series predictor evaluation. *Information Sciences, 191*, 192–213.
  https://doi.org/10.1016/j.ins.2011.12.028
- Cao, X., Zheng, S., Zhang, J., Chen, W., & Du, G. (2025). A hybrid CNN-Bi-LSTM
  model with feature fusion for accurate epilepsy seizure detection. *BMC Medical
  Informatics and Decision Making, 25*, 6.
  https://doi.org/10.1186/s12911-024-02845-0
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition
  Letters, 27*(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
- Guhdar, M., Mstafa, R. J., & Mohammed, A. O. (2025). Hybrid deep learning model
  for epileptic seizure classification by using 1D-CNN with multi-head attention
  mechanism. *arXiv preprint arXiv:2501.10342*.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical
  learning: Data mining, inference, and prediction* (2.ª ed.). Springer.
- Kashefi Amiri, H., Zarei, M., & Daliri, M. R. (2025). Epileptic seizure
  detection from electroencephalogram signals based on 1D CNN-LSTM deep learning
  model using discrete wavelet transform. *Scientific Reports, 15*, 32820.
  https://doi.org/10.1038/s41598-025-18479-9
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy
  estimation and model selection. *Proceedings of the 14th International Joint
  Conference on Artificial Intelligence (IJCAI)*, 1137–1143.
- LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). Efficient BackProp.
  En G. B. Orr & K.-R. Müller (Eds.), *Neural Networks: Tricks of the Trade*
  (pp. 9–50). Springer.
- Saha Tchinda, B., Tchiotsop, D., Djoufack Nkengfack, L. C., & Tchinda, R.
  (2025). Diagnosis of epileptic seizures from electroencephalogram signals using
  log-Mel spectrogram and a deep learning CNN model. *Heliyon, 11*(4), e42993.
  https://doi.org/10.1016/j.heliyon.2025.e42993
- Stone, M. (1974). Cross-validatory choice and assessment of statistical
  predictions. *Journal of the Royal Statistical Society: Series B, 36*(2),
  111–133.
- Varma, S., & Simon, R. (2006). Bias in error estimation when using
  cross-validation for model selection. *BMC Bioinformatics, 7*, 91.
  https://doi.org/10.1186/1471-2105-7-91
- Wang, X., Wang, G., Wu, T., Wang, Y., Kärkkäinen, T., & Cong, F. (2025).
  EEG-based seizure onset detection of frontal and temporal lobe epilepsies using
  1DCNN. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 33*,
  2263–2272. https://doi.org/10.1109/TNSRE.2025.3575900
- Wang, X., Wang, Y., Liu, D., Wang, Y., & Wang, Z. (2023). Automated recognition
  of epilepsy from EEG signals using a combining space–time algorithm of CNN-LSTM.
  *Scientific Reports, 13*, 14876. https://doi.org/10.1038/s41598-023-41537-z
- Yang, X., Zhao, J., Sun, Q., Lu, J., & Ma, X. (2021). An effective dual
  self-attention residual network for seizure prediction. *IEEE Transactions on
  Neural Systems and Rehabilitation Engineering, 29*, 1604–1613.
  https://doi.org/10.1109/TNSRE.2021.3103210

---

## Nota de alcance

Este documento es el plan. La implementación concreta (`src/cv.py`, refactor de
`compute_stats` y `train.py`, tests) se desarrolla en un paso siguiente, siguiendo
la sección 4 y los criterios de aceptación de la sección 6.
