# -*- coding: utf-8 -*-
"""
Generador del informe técnico (.docx) de la tesis.

Produce:
  - figures/diagrama_pipeline_informe.png   (flujo completo de datos)
  - figures/diagrama_arquitectura_informe.png (arquitectura CNN-1D)
  - figures/diagrama_gcp_informe.png        (estrategia operativa en GCP)
  - informe/Informe_Tecnico_CNN1D_CHB_MIT.docx

Uso:
    python scripts/generar_informe.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin display (Windows / servidores)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Cm, Inches, Pt, RGBColor

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "figures"
OUT_DIR = BASE_DIR / "informe"
OUT_FILE = OUT_DIR / "Informe_Tecnico_CNN1D_CHB_MIT.docx"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)   # azul institucional
DARK = "#333333"   # color para matplotlib (docx usa ACCENT/RGBColor)


# =============================================================================
# PARTE A — DIAGRAMAS
# =============================================================================

def _block(ax, x, y, w, h, text, fc="#EAF2F8", ec="#1F4E79", fs=8.5, weight="normal"):
    """Dibuja un bloque redondeado con texto centrado."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                         linewidth=1.1, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=DARK, wrap=True)


def _arrow(ax, x1, y1, x2, y2, color="#1F4E79", lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=lw, color=color,
                                 connectionstyle="arc3,rad=0"))


def diagrama_pipeline() -> Path:
    """Bloques del pipeline: ingesta -> clasificación -> alarma."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.4, 6.6))
    fig.patch.set_facecolor("white")

    # ----- Rama izquierda: preprocesado (offline) ----------------------------
    axl.set_xlim(0, 4.6); axl.set_ylim(0, 7.4); axl.axis("off")
    axl.set_title("Canal off-line de preprocesamiento", fontsize=10, color=DARK)

    bw, bh, x0 = 3.6, 0.5, 0.1
    steps_l = [
        ("EDF crudo CHB-MIT\n(24 pacientes, 256 Hz)", ""),
        ("Lectura MNE + reconstrucción\n16 canales bipolares (montaje TUEV)", ""),
        ("Filtro Butterworth pasabanda\n0.5–50 Hz (sosfiltfilt, fase cero)", ""),
        ("Ventaneo 5.12 s = 1310 muestras\noverlap 50% (stride 655)", ""),
        ("RobustScaler por canal\n(x − mediana)/IQR — stats de TRAIN", ""),
        ("CNN-1D entrenada\nlogit → sigmoide → p(crisis)", ""),
        ("Umbral τ (punto de operación de evento)", ""),
        ("Postprocesado de evento\n≥3 positivos en 4 ventanas + ≥30 s", ""),
        ("ALARMA DE CRISIS\n(1 por crisis larga)", ""),
    ]
    for i, (txt, _) in enumerate(steps_l):
        y = 7.4 - (i + 1) * 0.78
        first = i == 0
        _block(axl, x0, y, bw, bh, txt,
               fc="#FFF2CC" if first else "#EAF2F8",
               ec="#B8860B" if first else None,
               fs=8.3)
        if i < len(steps_l) - 1:
            _arrow(axl, x0 + bw / 2, y, x0 + bw / 2, y + 0.78)

    # ----- Rama derecha: entrenamiento / validación --------------------------
    axr.set_xlim(0, 4.6); axr.set_ylim(0, 7.4); axr.axis("off")
    axr.set_title("Rama de entrenamiento y validación", fontsize=10, color=DARK)

    steps_r = [
        ("split.json\n70/30 inter-paciente proporcional", ""),
        ("Train 13 / val 3 / test 8\n(split fijo, sin folds)", ""),
        ("Undersampling (ratio NEG_POS)\n+ toutes las positivas", ""),
        ("Ruido gaussiano en\nventanas de crisis (augmentation)", ""),
        ("BCEWithLogitsLoss + pos_weight\nADAMW + coseno + grad clip", ""),
        ("Selección de checkpoint\nmáx sens_evento s.a. falsas alarmas/h ≤ techo (val)", ""),
        ("Selección del mejor escenario\npor validación (select_best)", ""),
        ("Evaluación final sobre TEST\n(una sola vez)", ""),
    ]
    for i, (txt, _) in enumerate(steps_r):
        y = 7.4 - (i + 1) * 0.86
        _block(axr, x0, y, bw, bh, txt, fc="#E8F5E9" if i < 3 else "#EAF2F8",
               fs=8.3)
        if i < len(steps_r) - 1:
            _arrow(axr, x0 + bw / 2, y, x0 + bw / 2, y + 0.86)

    out = FIG_DIR / "diagrama_pipeline_informe.png"
    fig.tight_layout(w_pad=3)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def diagrama_arquitectura() -> Path:
    """Stack de la CNN-1D."""
    fig, ax = plt.subplots(figsize=(11.2, 4.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.set_title("Arquitectura SeizureCNN — CNN-1D (entrada (B, 16, 1310))",
                 fontsize=10, color=DARK)

    conv = [
        ("Conv1D 64@k7", "BN + ReLU\n+ MaxPool"),
        ("Conv1D 128@k5", "BN + ReLU\n+ MaxPool"),
        ("Conv1D 256@k5", "BN + ReLU\n+ MaxPool"),
        ("Conv1D 256@k3", "BN + ReLU\n+ MaxPool"),
        ("Conv1D 256@k3", "BN + ReLU\n+ MaxPool"),
        ("Conv1D 256@k3", "BN + ReLU\n+ MaxPool"),
    ]
    x = 0.3
    for i, (name, tail) in enumerate(conv):
        _block(ax, x, 2.2, 1.35, 1.5, name, fc="#EAF2F8", fs=8.0)
        _block(ax, x, 0.7, 1.35, 1.2, tail, fc="#F3E5F5", fs=7.6)
        if i < len(conv) - 1:
            _arrow(ax, x + 1.35, 3.0, x + 1.62, 3.0)
        x += 1.62

    _arrow(ax, x, 3.0, x + 0.28, 3.0)
    _block(ax, x + 0.28, 2.2, 1.5, 1.5, "GAP\n(1)", fc="#E8F5E9", fs=8.2)
    x += 1.78
    _arrow(ax, x, 3.0, x + 0.28, 3.0)
    _block(ax, x + 0.28, 2.2, 1.5, 1.5, "Dropout\n0.4", fc="#E8F5E9", fs=8.0)
    x += 1.78
    _arrow(ax, x, 3.0, x + 0.28, 3.0)
    _block(ax, x + 0.28, 2.2, 1.7, 1.5, "Dense 256\n+ ReLU + Dropout", fc="#E8F5E9", fs=7.8)
    x += 1.98
    _arrow(ax, x, 3.0, x + 0.28, 3.0)
    _block(ax, x + 0.28, 2.2, 1.4, 1.5, "Dense 1\n(logit)", fc="#FDECEC", fs=8.2)
    x += 1.68
    _arrow(ax, x, 3.0, x + 0.28, 3.0)
    _block(ax, x + 0.28, 2.2, 1.4, 1.5, "Sigmoid\np(crisis)", fc="#FDECEC", fs=8.2)

    out = FIG_DIR / "diagrama_arquitectura_informe.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def diagrama_gcp() -> Path:
    """Estrategia operativa en GCP con checkpoints."""
    fig, ax = plt.subplots(figsize=(11.4, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.set_title("Estrategia operativa en GCP: datos, cómputo y checkpoints",
                 fontsize=10, color=DARK)

    def box(x, y, w, h, txt, fc):
        _block(ax, x, y, w, h, txt, fc=fc, fs=8.0)

    # Fila arriba: repositorios persistentes
    box(0.2, 3.1, 3.6, 0.9, "Repo git\ncódigo versionado", "#FFF2CC")
    box(4.1, 3.1, 3.9, 0.9, "GCS Bucket (almacenamiento)\nEDF → train+val en SSD, test aparte", "#FFF2CC")
    box(8.4, 3.1, 3.8, 0.9, "GCS Bucket (artefactos)\ncheckpoints + histories, versionado", "#D6EAF8")

    # Fila del medio: cómputo
    box(2.2, 1.7, 3.6, 0.9, "Instancia GPU\n(VM / Vertex AI / Colab Pro)", "#E8F5E9")
    box(6.2, 1.7, 3.9, 0.9, "Entrenamiento\népocas → checkpoint .pt + history.json", "#EAF2F8")

    # Fila abajo: backup
    box(2.2, 0.1, 3.6, 0.9, "--backup-dir (copia tras cada mejora)\ngcsfuse so bre GCS / Drive", "#FBEEE6")
    box(6.2, 0.1, 3.9, 0.9, "Reanudación ante preempción\nreleer último checkpoint/history", "#FBEEE6")

    _arrow(ax, 2.0, 3.1, 2.2, 2.6)   # git -> instancia
    _arrow(ax, 4.8, 3.1, 3.6, 2.6)   # bucket datos -> instancia
    _arrow(ax, 7.2, 3.1, 7.2, 2.6)   # bucket artefactos -> entreno  (reviso)
    _arrow(ax, 7.2, 1.7, 4.0, 1.2)   # entrenamiento -> backup-dir
    _arrow(ax, 7.2, 0.1, 9.0, 1.2)   # backup-dir -> bucket artefactos (reutilizado)
    _arrow(ax, 2.2, 1.2, 2.2, 1.0)   # instancia -> backup
    _arrow(ax, 9.6, 1.2, 9.6, 1.0)   # bucket artefactos -> backup

    out = FIG_DIR / "diagrama_gcp_informe.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


# =============================================================================
# PARTE B — DOCX
# =============================================================================

def _setup(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.0)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for h, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)):
        st = doc.styles[h]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.color.rgb = ACCENT
        st.font.bold = True


def _p(doc, text, style=None, align=None):
    par = doc.add_paragraph(text, style=style)
    if align is not None:
        par.alignment = align
    return par


def _bullet(doc, text):
    doc.add_paragraph(text, style="List Bullet")


def _fig(doc, path: Path, width_in: float, caption: str):
    doc.add_picture(str(path), width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    _p(doc, caption, align=WD_ALIGN_PARAGRAPH.CENTER)


def _table(doc, headers, rows, widths=None):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        cell = tbl.rows[0].cells[j]
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            tbl.rows[i].cells[j].text = str(val)
    if widths:
        for j, w in enumerate(widths):
            for i in range(len(rows) + 1):
                tbl.rows[i].cells[j].width = Cm(w)
    return tbl


def build() -> Path:
    doc = Document()
    _setup(doc)

    # ---------- Portada -------------------------------------------------------
    for _ in range(4):
        _p(doc, "")
    _p(doc, "Detección Automática de Crisis Epilépticas en Señales EEG\nmediante Red Neuronal Convolucional 1D",
        style="Title")
    _p(doc, "")
    _p(doc, "Informe Técnico de Tesis — Trabajo Final de Carrera",
        align=WD_ALIGN_PARAGRAPH.CENTER)
    _p(doc, "")
    _p(doc, "Pipeline completo, decisiones técnicas y científicas, y estrategia\noperativa de cómputo en la nube (GCP)",
        align=WD_ALIGN_PARAGRAPH.CENTER)
    _p(doc, "")
    _p(doc, "Dataset: CHB-MIT Scalp EEG Database (PhysioNet)",
        align=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_page_break()

    # ---------- Resumen ejecutivo --------------------------------------------
    doc.add_heading("Resumen Ejecutivo", level=1)
    _p(doc, "Este informe documenta el diseño, la implementación y la estrategia operativa de "
            "un sistema de apoyo a la decisión clínica capaz de detectar crisis epilépticas en "
            "registros continuos de electroencefalograma (EEG). El sistema clasifica ventanas "
            "temporales de señal cruda mediante una CNN-1D y traduce sus predicciones a alarmas "
            "a nivel de crisis (evento), que es la unidad relevante para el personal médico.")
    _p(doc, "El trabajo se estructura en tres planos. El primero es el pipeline de datos: de los "
            "archivos EDF crudos del dataset público CHB-MIT se reconstruyen 16 canales bipolares "
            "del montaje TUEV, se filtra la señal con un pasabanda de fase cero de 0.5 a 50 Hz, se "
            "segmenta en ventanas de 5.12 s con 50% de solapamiento y se normaliza por canal con "
            "un escalador robusto cuyas estadísticas se estiman exclusivamente sobre el conjunto de "
            "entrenamiento (para evitar fuga de datos).")
    _p(doc, "El segundo plano es la metodología de validación: un split inter-paciente 70/30 con "
            "repaso proporcional en conjuntos fijos de entrenamiento (13 pacientes), validación "
            "(3 pacientes) y test (8 pacientes), sin validación cruzada para acotar el costo "
            "computacional. Cada escenario de hiperparámetros se entrena una única vez, se "
            "selecciona el mejor usando la validación y el test permanece intacto hasta la "
            "evaluación final. La selección del mejor modelo no se hace por pérdida bruta ni por "
            "métricas de ventana, sino por el criterio clínico de minimizar la tasa de falsas "
             "alarmas de evento por hora sujeto a detectar al menos el 90% de las crisis.")
    _p(doc, "El tercer plano es la operación: se describe una estrategia reproducible de cómputo "
            "en Google Cloud (GCP) con checkpoints respaldados de forma incremental, versionado de "
            "experimentos y recuperación ante interrupciones, de modo que ninguna corrida de "
            "entrenamiento se pierda por la naturaleza efímera de las instancias virtuales.")

    # ---------- 1. Introducción -----------------------------------------------
    doc.add_heading("1. Introducción", level=1)
    doc.add_heading("1.1 Contexto clínico", level=2)
    _p(doc, "La epilepsia es una de las afecciones neurológicas más prevalentes del mundo y una "
            "proporción relevante de sus pacientes no responde a la medicación antiepiléptica. "
            "En los monitoreos continuos de larga duración (por ejemplo, en unidades de "
            "telemetría de video-EEG), el recurso habitual de detección de crisis es la "
            "inspección visual del registro por parte de neurólogos. Este proceso es lento, "
            "costoso y altamente susceptible a la fatiga: en registros de varias horas o días, "
            "una crisis poco frecuente puede permanecer sin detectar. Esa es la motivación "
            "clínica: un sistema automático que filtre las horas de registro y alerte al personal "
            "ante posibles crisis, independientemente de la atención humana continua.")
    doc.add_heading("1.2 Objetivo", level=2)
    _p(doc, "El objetivo del trabajo es implementar y validar una arquitectura de inteligencia "
            "artificial — una red neuronal convolucional unidimensional (CNN-1D) — que clasifique "
            "ventanas temporales de señal EEG cruda como crisis (ictal) o no crisis (basal), y que "
            "colapse esas predicciones en alarmas a nivel de crisis con un nivel de falsas "
            "alarmas clínicamente aceptable.")
    doc.add_heading("1.3 Alcance (formato MVP)", level=2)
    _p(doc, "La entrega corresponde a un producto mínimo viable (MVP): un conjunto de scripts de "
            "línea de comandos que permiten preprocesar el dataset, entrenar el modelo, comparar "
            "experimentos y evaluar e inferir sobre archivos EDF nuevos. Quedan fuera del alcance "
            "las interfaces gráficas, las bases de datos relacionales, las representaciones "
            "tiempo-frecuencia (espectrogramas / CNN-2D) y el transfer learning, que se documentan "
            "como trabajo futuro.")

    # ---------- 2. Marco conceptual y estado del arte ------------------------
    doc.add_heading("2. Marco conceptual y estado del arte", level=1)
    _p(doc, "La literatura de detección automática de crisis evalúa los sistemas en dos niveles "
            "complementarios. El nivel de segmento (ventana) cuenta aciertos de ventanas; el nivel "
            "de evento (crisis completa) cuenta si el sistema emitió una alarma durante la crisis. "
            "El estándar clínico moderno es el segundo. Wang et al. (IEEE TNSRE, 2025), en su "
            "artículo de referencia que emplea una CNN-1D sobre EEG de epilepsia frontal y "
            "temporal, reportan como métricas primarias la sensibilidad a nivel de evento, la "
             "tasa de falsas alarmas por hora y la latencia, y como métricas secundarias "
            "las de segmento (sensibilidad, especificidad, accuracy y AUC).")
    _p(doc, "Para la elección del punto de operación, la teoría de análisis ROC de Fawcett "
            "(2006) establece que la curva ROC describe el trade-off entre sensibilidad y "
            "especificidad a nivel de muestra, y que el punto de operación debe elegirse según el "
            "costo de cada error. En detección de crisis, ese costo no es simétrico: perder una "
            "crisis (falso negativo) es mucho más grave que tolerar una alarma falsa esporádica. "
            "Por ello, la propuesta del presente trabajo fija un requisito clínico explícito "
            "(detectar ≥ 90% de las crisis) y, dentro de los umbrales que lo cumplen, minimiza el "
             "la tasa de falsas alarmas por hora, en línea con la exigencia de la propuesta de tesis de reportar la tasa "
            "de falsas alarmas por unidad de tiempo.")
    _p(doc, "Otros trabajos recientes del marco teórico (Wang et al., Sci. Rep. 2023; Saha et "
            "al., Heliyon 2025; Kashefi et al., Sci. Rep. 2025) operan sobre detección a nivel de "
            "segmento con métricas convencionales (sensibilidad, especificidad, precisión, AUC). "
            "Ninguno de ellos describe formalmente la selección del umbral por sensibilidad a "
            "nivel de evento. Ese hueco metodológico es precisamente lo que este trabajo aborda de "
            "forma explícita y documentada: el criterio de selección del modelo es una decisión "
            "operativa propia, alineada con la métrica que percibe el clínico (cuántas crisis "
            "detectadas, cuántas falsas alarmas por hora).")

    # ---------- 3. Datos ------------------------------------------------------
    doc.add_heading("3. Datos clínicos: el dataset CHB-MIT", level=1)
    doc.add_heading("3.1 Características del dataset", level=2)
    _p(doc, "Se utiliza el CHB-MIT Scalp EEG Database (PhysioNet, versión 1.0.0), un corpus "
            "público de registros de pacientes pediátricos con epilepsia refractaria. El dataset "
            "comprende 24 pacientes (chb01 a chb24), cerca de 900 archivos EDF de una hora "
            "aproximadamente y unos 43 GB de datos, muestreados a 256 Hz con montajes que incluyen "
            "hasta 23 canales de EEG.")
    _p(doc, "Una decisión de diseño fundamental es la elección del montaje de referencia: los 16 "
            "canales bipolares del estándar TUEV (TUH EEG Corpus). Un canal bipolar es la resta de "
            "dos electrodos vecinos (por ejemplo FP1-F7 = FP1 − F7). La resta cancela cualquier "
            "señal común a ambos electrodos (ruido de referencia, deriva de continua) y resalta la "
            "actividad eléctrica local de la pareja. Esta es la razón por la que la mayor parte "
            "de los canales de los EDFs de CHB-MIT están configurados de forma bipolar y por la "
            "que los canales que no lo están deben reconstruirse por resta (más adelante se "
            "detalla la estrategia de reconstrucción).")
    doc.add_heading("3.2 Anotaciones", level=2)
    _p(doc, "Cada paciente incluye un archivo chbXX-summary.txt que lista, por cada EDF, los "
            "intervalos de crisis: tiempo de inicio y fin en segundos. El parser de anotaciones "
            "debe soportar dos sintaxis presentes en el corpus (con y sin numeración de la "
            "crisis), y descartar intervalos inconsistentes. El resultado es un mapa "
            "paciente → archivo → lista de intervalos [inicio, fin], que se usa tanto para el "
            "etiquetado de ventanas como para el cálculo de métricas a nivel de evento.")
    doc.add_heading("3.3 Desbalance de clases", level=2)
    _p(doc, "La crisis ocupa una fracción muy pequeña del registro: alrededor del 0.5% de las "
            "ventanas del conjunto de entrenamiento son positivas. Este desbalance extremo "
            "condiciona todas las decisiones posteriores: la función de pérdida, la estrategia de "
            "muestreo, el criterio de selección del modelo y las métricas de reporte. Un "
            "clasificador que prediciera siempre “no crisis” alcanzaría una accuracy superior al "
            "99%, pero sería clínicamente inútil. Por eso la accuracy sola nunca es un criterio "
            "válido en este dominio.")

    # ---------- 4. Preprocesamiento ------------------------------------------
    doc.add_heading("4. Pipeline de preprocesamiento y diagrama de bloques", level=1)
    _p(doc, "La figura siguiente sintetiza el flujo completo del sistema, desde la ingesta de los "
            "registros hasta la emisión de alarmas, junto con la rama de entrenamiento y "
            "validación.")
    _fig(doc, diagrama_pipeline(), 6.5, "Figura 1 — Diagrama de bloques del sistema completo.")

    doc.add_heading("4.1 Lectura del EDF y reconstrucción de canales", level=2)
    _p(doc, "Cada EDF se lee con MNE-Python. El montaje no es uniforme entre pacientes (incluso "
            "chb17 tiene sub-montajes a/b/c y chb24 utiliza un montaje particular), por lo que "
            "los 16 canales TUEV no siempre vienen pre-armados. La estrategia es construir un "
            "plan de reconstrucción por archivo:")
    _bullet(doc, "Si el canal bipolar buscado ya existe en el EDF, se copia directamente su fila.")
    _bullet(doc, "Si existen dos electrodos únicos (sin referencia visible), el canal se "
                 "reconstruye como la resta de ambos.")
    _bullet(doc, "Si los electrodos están referidos a una referencia común (por ejemplo CS2), el "
                 "canal se reconstruye restando dos canales referidos que comparten esa "
                 "referencia: (A−REF) − (B−REF) = A−B. La referencia se cancela algebraicamente.")
    _p(doc, "Los archivos cuya frecuencia de muestreo difiere de 256 Hz o cuyo montaje no permite "
            "la reconstrucción se descartan con un motivo registrado, garantizando trazabilidad.")
    doc.add_heading("4.2 Filtrado pasabanda con fase cero", level=2)
    _p(doc, "La señal se filtra con un Butterworth pasabanda de 0.5 a 50 Hz. La banda de 0.5 Hz "
            "elimina la deriva lenta de la línea de base (respiración, movimiento de electrodo), "
            "mientras que 50 Hz elimina el ruido de red y el contenido de alta frecuencia no "
            "relacionado con la actividad ictal. El orden 5 es un compromiso entre pendiente del "
            "corte y estabilidad numérica; al aplicar sosfiltfilt, el filtro se aplica en ambas "
            "direcciones y la respuesta en magnitud queda al cuadrado, por lo que el orden "
            "efectivo es 10.")
    _p(doc, "El filtrado se hace con sosfiltfilt (fase cero), no causal. Esto tiene dos "
            "consecuencias rigurosamente cuantificadas: no agrega latencia temporal (el retardo de "
            "grupo es cero) y utiliza información del futuro únicamente en los bordes de cada "
            "archivo (~0.36% del registro de una hora), donde el transitorio se centra en el borde "
            "en lugar de acumularse. Toda la latencia del sistema proviene, por lo tanto, de la "
            "duración de la ventana analizada por la red y no del filtro.")

    doc.add_heading("4.3 Segmentación en ventanas", level=2)
    _p(doc, "La señal se segmenta en ventanas nominales de 5.12 s. A 256 Hz se obtienen "
            "int(5.12 × 256) = 1310 muestras, con una duración efectiva de 1310/256 = "
            "5.1171875 s. El redondeo hacia abajo garantiza un número par de muestras, lo que "
            "permite un solapamiento exacto del 50% y un pooling de MaxPool(2) sin restos. El "
            "paso (stride) es de 655 muestras (1310/2), de modo que cada ventana comparte el 50% "
            "de su contenido con la contigua. El solapamiento aporta continuidad temporal: una "
            "crisis que comienza entre dos ventanas genera una serie de ventanas positivas "
            "consecutivas, algo que explota el postprocesado a nivel de evento.")
    doc.add_heading("4.4 Etiquetado binario", level=2)
    _p(doc, "Una ventana se etiqueta como crisis (1) si solapa cualquier porción de un intervalo "
            "anotado (criterio de intervalo: start < end_crisis y end > start_crisis), y como no "
            "crisis (0) en caso contrario. El etiquetado se hace tanto al construir el índice "
            "global del dataset (aprovechando solo las cabeceras de los EDF) como al procesar "
            "cada archivo, con verificación cruzada de consistencia entre ambos.")

    doc.add_heading("4.5 Escalado robusto por canal", level=2)
    _p(doc, "La entrada de la red se normaliza con un RobustScaler por canal: x' = (x − "
            "mediana_c)/IQR_c, donde IQR es el rango intercuartílico (percentil 75 menos 25). "
            "Las razones teóricas son tres:")
    _bullet(doc, "Aceleración de convergencia: el gradiente de una capa es proporcional a la "
                 "magnitud de su entrada (LeCun et al., 1998). Sin normalización, los canales de "
                 "mayor amplitud dominarían la actualización de pesos.")
    _bullet(doc, "Invariabilidad a la ganancia: la amplitud absoluta del EEG (en microvoltios) "
                 "varía entre pacientes, sesiones e impedancias. Lo que define un patrón ictal es "
                 "la forma de la onda, no su escala. Restar mediana y dividir por IQR hace al "
                 "modelo invariable a la ganancia global del canal.")
    _bullet(doc, "Robustez a los artefactos: la mediana y el IQR son estadísticos de orden que "
                 "no se corren con picos extremos aislados (parpadeos, artefactos musculares), a "
                 "diferencia de la media/desvío (z-score) o del mínimo/máximo (min-max).")
    _p(doc, "Las estadísticas (mediana e IQR por canal) se estiman únicamente sobre las ventanas "
            "del conjunto de entrenamiento y se persisten en un archivo .npz que viaja dentro del "
            "checkpoint. Validación, test e inferencia reutilizan exactamente esas estadísticas. "
            "Esta es la regla anti-fuga de datos (data leakage): si las estadísticas se "
            "calcularan con datos de test, el modelo recibiría información de test ya procesada y "
            "las métricas finales serían falsamente optimistas.")
    _p(doc, "Dado que el conjunto de train completo rondaría los 14.000 millones de valores, las "
            "estadísticas se estiman con un submuestreo uniforme aleatorio (semilla fija): 1 de "
            "cada 16 ventanas y, dentro de ellas, 1 de cada 16 muestras (~55 millones de puntos, "
            "220 MB). Como mediana e IQR son estadísticos robustos, el submuestreo uniforme no "
            "los sesga y converge al valor exacto del conjunto completo. El detalle de cuántos "
            "archivos, ventanas y puntos se usaron se guarda junto con las estadísticas para "
            "auditoría.")

    doc.add_heading("4.6 Generación de ventanas al vuelo", level=2)
    _p(doc, "Las ventanas no se materializan en disco. El dataset mantiene un índice global "
            "ventana → (archivo, ventana local) construido una sola vez con las cabeceras de los "
            "EDF, y procesa cada archivo recién cuando el DataLoader lo pide por primera vez, "
            "con una caché de un archivo en RAM (~1 s de procesamiento por archivo de una hora). "
            "Esta es una decisión de ingeniería: precargar el train completo en float32 "
            "significaría unos 47 GB adicionales de almacenamiento que deben regenerarse cada "
            "vez que cambie el filtro, el ventaneo o el escalado. El patrón es el estándar de la "
            "literatura EEG-DL (Braindecode) y mantiene la misma matemática que el "
            "pre-armado, sin el costo de disco.")

    # ---------- 5. Diseño experimental --------------------------------------
    doc.add_heading("5. Diseño experimental y validación", level=1)
    doc.add_heading("5.1 Split inter-paciente (70/30 proporcional)", level=2)
    _p(doc, "La unidad de asignación es el paciente completo, nunca la ventana. Dos ventanas del "
            "mismo paciente pueden ser muy parecidas (sobre todo con 50% de solapamiento), por lo "
            "que separar por ventanas induciría una fuga de información y sobreestimaría la "
            "generalización. Separar por paciente mide lo que realmente interesa: qué sucede "
            "cuando el sistema encuentra a un paciente que nunca vio.")
    _p(doc, "El reparto es proporcional. El algoritmo ordena los pacientes por severidad "
            "(segundos de crisis) y los asigna de forma codiciosa a train o test según cuál "
            "opción deje las fracciones acumuladas más cercanas al objetivo (train ≈ 70%). La "
            "bondad se mide con dos métricas: la fracción de archivos (~horas de registro) y la "
            "fracción de segundos de crisis (~ventanas positivas), ponderándose los segundos de "
            "crisis con peso 2.0 (SPLIT_W_ZSEC) porque son los que determinan cuántas ventanas "
            "positivas quedan en test y, por lo tanto, la fiabilidad de la sensibilidad medida. "
            "Con peso 1.0 el test quedaba con solo ~13% de las crisis; con 2.0 se logra ~31% de "
            "archivos y ~19% de crisis, minimizando el peor desvío frente al 30% objetivo.")
    _p(doc, "Dentro del conjunto de train se reservan 3 pacientes para validación, elegidos en "
            "posiciones centrales de la distribución de severidad para que sean representativos.")
    doc.add_heading("5.2 Selección de hiperparámetros por validación fija", level=2)
    _p(doc, "La elección de hiperparámetros no debe usar el test. Dado el costo computacional de "
            "entrenar una CNN por escenario, se opta por un split fijo en lugar de una validación "
            "cruzada: el conjunto de desarrollo (16 pacientes, train + val) se reparte de una "
            "única vez en 13 de entrenamiento y 3 de validación, y el test original (8 pacientes) "
            "permanece intacto. Cada escenario de hiperparámetros se entrena una sola vez sobre "
            "los 13 pacientes de train y se compara por sus métricas de validación. Esta es la "
            "concesión metodológica del MVP: se sacrifica la estimación de la variabilidad entre "
            "particiones (que aportaría una validación cruzada agrupada por paciente) a cambio de "
            "reducir de forma drástica el número de entrenamientos, viable con los recursos "
            "disponibles.")
    _p(doc, "El escalador robusto se calcula una única vez sobre los 13 pacientes de train y se "
            "comparte entre todos los escenarios, respetando la regla anti-leakage: validación y "
            "test reutilizan exactamente esas estadísticas.")
    doc.add_heading("5.3 El test como examen final", level=2)
    _p(doc, "El test evalúa el modelo final una única vez, después de fijar arquitectura, "
            "preprocesado, hiperparámetros, umbral y postprocesado. Si se modificara algo tras "
            "mirar el test, dejaría de ser una medición independiente. Este es el protocolo "
            "clásico para evitar el sobreajuste a la validación y al test.")

    # ---------- 6. Arquitectura ----------------------------------------------
    doc.add_heading("6. Arquitectura de la red neuronal (CNN-1D)", level=1)
    _fig(doc, diagrama_arquitectura(), 6.5, "Figura 2 — Arquitectura SeizureCNN.")
    _p(doc, "La red (SeizureCNN) clasifica cada ventana (16 canales × 1310 muestras) en un logit "
            "de crisis. Está compuesta por 6 bloques convolucionales "
            "Conv1d + BatchNorm + ReLU + MaxPool, un Global Average Pooling, dropout y dos capas "
            "densas. Los filtros crecen (64 → 128 → 256) mientras el tamaño de kernel decrece "
            "(7 → 5 → 3). Estas dos tendencias son complementarias: los primeros bloques ven "
            "contexto temporal largo y construyen representaciones de bajo nivel; los últimos "
            "refinan detalles finos y combinan las abstracciones de los anteriores, por lo que "
            "necesitan más canales de salida.")
    _p(doc, "Justificación teórica de cada componente:")
    _bullet(doc, "Convolución 1D: el filtro aprendido se desliza sobre el tiempo buscando "
                 "patrones locales (picos rítmicos, ondas agudas) independientemente de su "
                 "posición exacta en la ventana. El padding preserva la longitud temporal.")
    _bullet(doc, "BatchNorm: normaliza las activaciones por canal, estabilizando el "
                 "entrenamiento y reduciendo el cambio de la distribución de entrada de cada "
                 "capa (internal covariate shift; Ioffe y Szegedy, 2015). Los parámetros "
                 "gamma/beta devuelven a la red la libertad de reescalar cada canal.")
    _bullet(doc, "ReLU: introduce no linealidad; sin ella, apilar capas lineales es "
                 "equivalente a una única capa lineal (sin ganancia de representación).")
    _bullet(doc, "MaxPool(2): reduce a la mitad la longitud temporal, baja el cómputo y añade "
                 "cierta invariancia a pequeños desplazamientos temporales.")
    _bullet(doc, "Global Average Pooling: colapsa cada canal temporal a su promedio, reduciendo "
                 "drásticamente el número de parámetros frente a una capa densa sobre toda la "
                 "secuencia y actuando como regularizador.")
    _bullet(doc, "Dropout (0.4): durante el entrenamiento apaga al azar el 40% de las neuronas "
                 "del vector de características, obligando a la red a aprender patrones "
                 "redundantes y reduciendo el sobreajuste.")
    _p(doc, "El conteo de parámetros se mantiene en el orden de 0.5 a 2 millones, entrenable en "
            "una sola GPU (por ejemplo una T4/L4 de Colab Pro) en tiempos razonables. A modo de "
            "referencia, el smoke-test de arquitectura reporta ~9.35 GB de multiplicaciones por "
            "batch de 64, dato que se utilizará en la comparación futura de costo computacional "
            "frente a una CNN-2D.")

    # ---------- 7. Entrenamiento ---------------------------------------------
    doc.add_heading("7. Entrenamiento del modelo", level=1)
    doc.add_heading("7.1 Manejo del desbalance: undersampling y pos_weight", level=2)
    _p(doc, "El desbalance extremo se ataca en dos frentes durante el entrenamiento. Por un lado, "
            "el muestreo: cada época conserva todas las ventanas positivas y una muestra aleatoria "
            "de NEG_POS_RATIO = 3 negativas por positiva (sin reemplazo), con global shuffle en "
            "memoria. Por otro lado, la pérdida: se usa la Binary Cross Entropy con logits "
            "ponderada, BCEWithLogitsLoss(pos_weight), donde el error de clasificar mal una crisis "
            "se multiplica por un factor equivalente al ratio efectivamente muestreado. La fórmula "
            "es L = −[pos_weight·y·log(σ(z)) + (1−y)·log(1−σ(z))].")
    _p(doc, "La validación y el test NO se tocan: se evalúa la distribución natural completa, sin "
            "undersampling y sin pos_weight. Esta es la condición necesaria para que la "
             "especificidad, el FPR y la tasa de falsas alarmas por hora sean honestos: solo midiendo sobre la "
            "distribución real de negativos se puede cuantificar cuántas falsas alarmas generaría "
            "el sistema por hora de registro.")
    doc.add_heading("7.2 Optimizador, programador y regularización", level=2)
    _bullet(doc, "AdamW: optimizador Adam con weight decay desacoplado del momento adaptativo. "
                 "El decay castiga los pesos grandes (regularización L2), previniendo que la red "
                 "memorice ruido específico del train, sin interferir con la tasa adaptativa por "
                 "peso.")
    _bullet(doc, "Scheduler coseno: baja el learning rate base de 1e-4 a ~0 siguiendo una curva "
                 "coseno a lo largo del entrenamiento. Estabiliza las épocas finales, donde con "
                 "lr alto el modelo oscilaría entre “detectar todo” y “no detectar nada”.")
    _bullet(doc, "Gradient clipping (norma L2 = 1.0): si la norma total del gradiente supera el "
                 "tope, todos los gradientes se escalan proporcionalmente. Es un límite de "
                 "velocidad que evita que un batch ruidoso produzca un salto de pesos "
                 "catastrófico (el gradiente siguiendo su dirección, solo con tamaño acotado).")
    _bullet(doc, "Semilla global (SEED = 42): fija el azar de Python, NumPy, PyTorch y cuDNN "
                 "(modo determinista), de modo que dos corridas idénticas produzcan exactamente "
                 "los mismos splits, batches y pesos iniciales. Es un requisito de "
                 "reproducibilidad científica.")
    doc.add_heading("7.3 Criterio de early stopping y guardado", level=2)
    _p(doc, "El early stopping y la selección del checkpoint no usan la pérdida bruta de "
            "validación (ver sección siguiente), sino el criterio clínico a nivel de evento: se "
             "guarda el modelo cuando el punto de operación mejora (mayor sensibilidad de crisis) "
             "respetando un techo de falsas alarmas por hora. Los checkpoints (formato versionado)"
            " contienen pesos, configuración de arquitectura, estadísticas del escalador, umbral "
            "operativo, configuración de evento e hiperparámetros del escenario, todo serializado "
            "de forma segura y auditable.")

    doc.add_heading("7.4 Augmentación con ruido gaussiano", level=2)
    _p(doc, "Siguiendo a Wang et al. (IEEE TNSRE, 2025), que agrega ruido gaussiano a las "
            "muestras ictales para aliviar el desbalance de clases, se suma ruido aditivo "
            "N(0, σ) exclusivamente a las ventanas de crisis durante el entrenamiento. El desvío "
            "σ (NOISE_STD = 0.1) está expresado en unidades de la señal ya escalada por el "
            "RobustScaler (mediana ~0, IQR ~1), es decir, relativo a la dispersión del canal y no "
            "una amplitud absoluta en microvoltios. La augmentación diversifica la clase "
            "minoritaria en cada época sin tocar la clase basal, y puede desactivarse con "
            "NOISE_STD = 0. Se aplica con un generador aleatorio independiente del muestreo, de "
            "modo que undersampling y shuffle permanecen reproducibles con la misma semilla.")

    # ---------- 8. Métricas y criterio de selección --------------------------
    doc.add_heading("8. Métricas y criterio de selección del modelo", level=1)
    doc.add_heading("8.1 Métricas a nivel de segmento (ventana)", level=2)
    _p(doc, "Sobre la matriz de confusión (TP, FP, TN, FN) de cada umbral se calculan las "
            "métricas clásicas de clasificación binaria:")
    _table(doc,
           ["Métrica", "Fórmula", "Interpretación"],
           [
               ["Sensibilidad (Recall)", "TP / (TP + FN)", "Fracción de ventanas de crisis detectadas"],
               ["Especificidad", "TN / (TN + FP)", "Fracción de ventanas normales correctamente descartadas"],
               ["Tasa de falsos positivos (FPR)", "FP / (TN + FP) = 1 − Espec.", "Probabilidad de marcar una ventana normal como crisis"],
               ["FP por hora", "FP / horas cubiertas", "Falsos positivos de ventana por hora registrada"],
               ["Accuracy", "(TP + TN) / total", "Acierto global; engañosa bajo desbalance"],
               ["F1", "2·P·R / (P + R)", "Media armónica de precisión y recall"],
               ["MCC", "(TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))", "Correlación binaria balanceada; robusta al desbalance"],
           ],
           widths=[4.2, 5.6, 6.2])
    _p(doc, "El MCC (coeficiente de correlación de Matthews) resume toda la matriz de confusión "
            "en un único coeficiente entre −1 y +1 y es robusto al desbalance de clases, a "
            "diferencia de la accuracy. El F1 pondera falsos positivos y falsos negativos por "
            "igual, lo que en dominio clínico subestima el costo del falso negativo; por eso se "
            "reporta junto con el resto, nunca como criterio único.")
    doc.add_heading("8.2 Métricas a nivel de evento (crisis)", level=2)
    _p(doc, "El clínico no cuenta ventanas: cuenta crisis y alarmas. A nivel de evento se "
            "definen tres magnitudes:")
    _bullet(doc, "Sensibilidad de evento = crisis detectadas / crisis totales. Una crisis está "
                 "detectada si existe una alarma dentro de [onset, onset + 30 s] (EVENT_MAX_LATENCY).")
    _bullet(doc, "Tasa de falsas alarmas de evento por hora = falsas alarmas / horas de registro cubiertas. Es "
                 "el “falsos positivos por hora” a nivel de evento; se distingue de FDR, que normalmente "
                 "divide falsas alarmas por el total de alarmas. "
                 "y de los sistemas comerciales.")
    _bullet(doc, "Latencia = tiempo desde el onset de la crisis hasta la primera alarma "
                 "(media y mediana).")
    _p(doc, "El postprocesado que permite pasar de ventanas a eventos es una regla de persistencia "
            "temporal: se dispara una alarma cuando hay al menos POSITIVES_FOR_EVENT = 3 "
            "predicciones positivas dentro de las últimas WINDOW_RANGE_FOR_EVENT = 4 ventanas "
            "consecutivas. Esta regla filtra positivos aislados (un parpadeo, un artefacto de un "
            "único ventana) y colapsa la ráfaga de ~15 ventanas que genera una crisis de 40 s en "
            "una sola alarma. Un intervalo mínimo de 30 s entre alarmas evita contar dos veces "
            "la misma crisis larga.")
    _p(doc, "Las alarmas se ubican en el tiempo de decisión, definido como el final de la "
            "ventana: d_t = t + 5.117 s. La red consume la ventana completa antes de emitir su "
            "logit; no tiene sentido atribuir la predicción al inicio de la ventana. Esta "
            "convención temporal única se centraliza en el módulo de tiempos y es la que se "
            "usa tanto en la evaluación como en la inferencia, de modo que latencias y "
            "ordenamiento de alarmas son siempre consistentes.")
    doc.add_heading("8.3 El criterio de selección: sensibilidad de evento, no val_loss", level=2)
    _p(doc, "La decisión metodológica central del trabajo es elegir el mejor checkpoint por el "
            "criterio clínico a nivel de evento, no por la pérdida de validación ni por métricas "
            "de ventana. La justificación es doble.")
    _p(doc, "En primer lugar, la pérdida bruta (BCE sin pos_weight) sobre la distribución natural "
            "está dominada por las ventanas negativas: el clasificador trivial “siempre no "
            "crisis” obtiene una pérdida bajísima y cualquier modelo que valore la clase rara "
            "parece “peor”. Este es un resultado conocido para problemas con desbalance extremo "
            "(He y García, 2009; Branco et al., 2016). En corridas reales, el criterio de "
            "val_loss seleccionó un modelo conservador (sensibilidad de ventana 0.23 y más de 12 "
            "falsos positivos por hora), inútil clínicamente.")
    _p(doc, "En segundo lugar, optimizar la sensibilidad a nivel de ventana es un proxy "
            "engañoso. Para lograr una sensibilidad de ventana alta el modelo debe etiquetar "
            "correctamente las ventanas de los bordes de cada crisis, donde la actividad ictal "
            "es incipiente; la única forma de hacerlo es bajar mucho el umbral, disparando los "
            "falsos positivos por hora a valores clínicamente inviables (en nuestras corridas, "
            "sensibilidad de ventana 0.8 costó unas 280 falsas alarmas por hora). En cambio, "
            "para detectar una crisis completa basta con una alarma durante su curso: un modelo "
            "con sensibilidad de ventana “solo” 0.83 ya detectó 16 de 16 crisis (sensibilidad de "
            "evento 1.0) en la validación.")
    _p(doc, "El criterio adoptado se formaliza como un problema de optimización con restricción:")
    _p(doc, "op = argmax_τ Sens_evento(τ), sujeto a Falsas_alarmas_evento_por_hora(τ) ≤ techo", align=WD_ALIGN_PARAGRAPH.CENTER)
    _p(doc, "donde τ recorre una malla de umbrales (0.05…0.95). El techo clínico de falsas alarmas "
            "por hora (MAX_FALSE_ALARMS_PER_HOUR = 10) fija una tasa de alarma tolerable y, dentro de "
            "los umbrales que lo respetan, se maximiza la sensibilidad de crisis. El umbral elegido "
            "(op_threshold), la sensibilidad y la tasa de falsas alarmas por hora del punto de "
            "operación se guardan en el checkpoint. La pérdida de validación se sigue registrando "
            "solo como referencia de diagnóstico.")
    _p(doc, "Una aclaración metodológica importante: este criterio es una elección operativa de "
            "este trabajo. El análisis del marco teórico muestra que los papers de referencia "
            "evalúan ambos niveles simultáneamente (TNSRE 2025) o se mantienen a nivel de "
            "segmento (Fawcett 2006 y los trabajos Sci. Rep. 2023/2025), pero ninguno formaliza "
            "la selección del umbral por sensibilidad de evento. La regla aquí adoptada es, por "
            "lo tanto, una contribución metodológica propia, alineada con la métrica de falsas "
            "alarmas por unidad de tiempo exigida en la propuesta de tesis y con la realidad "
            "perceptiva del clínico.")

    # ---------- 9. Protocolo de experimentación ------------------------------
    doc.add_heading("9. Protocolo completo de experimentación", level=1)
    _p(doc, "El flujo operativo distingue dos unidades: el escenario (una configuración de "
            "hiperparámetros) y el modelo ganador (el checkpoint del escenario elegido por "
            "validación). Cada escenario se entrena una única vez.")
    doc.add_heading("9.1 Entrenamiento de los escenarios", level=2)
    _p(doc, "Cada escenario se entrena una única vez con la misma configuración (ratio, "
            "pos_weight, seed) y el mismo escalador (calculado sobre los 13 pacientes de train). "
            "Cada corrida produce un checkpoint .pt y un historial .history.json con las métricas "
            "por época. Los nombres incluyen los hiperparámetros (por ejemplo "
            "models/ratio3_pw1.pt) para no pisar resultados.")
    doc.add_heading("9.2 Selección del mejor escenario (select_best)", level=2)
    _p(doc, "Al terminar los escenarios, select_best lee los historiales y compara el punto de "
            "operación de cada uno en validación. Un escenario es elegible si guardó checkpoint y "
            "su tasa de falsas alarmas de evento por hora respeta el techo clínico. Entre los "
             "elegibles se elige el de mayor sensibilidad de crisis en validación. Nunca se usa test en esta etapa.")
    doc.add_heading("9.3 Evaluación final sobre test", level=2)
    _p(doc, "El checkpoint ganador se evalúa sobre el test una sola vez. El resultado de ese "
            "comando es el resultado final del MVP. Si después se cambia un hiperparámetro, hay "
            "que repetir la selección por validación y volver a reservar el test.")

    # ---------- 10. Estrategia operativa en GCP ------------------------------
    doc.add_heading("10. Estrategia operativa de cómputo en GCP (con checkpoints)", level=1)
    _fig(doc, diagrama_gcp(), 6.5, "Figura 3 — Estrategia operativa en GCP.")
    doc.add_heading("10.1 Principio rector: separar estado efímero de estado durable", level=2)
    _p(doc, "Las instancias de cómputo en la nube (y las sesiones de Colab) tienen discos "
            "efímeros que se pierden al terminar la sesión. La estrategia operativa se basa en "
            "mantener en almacenamiento persistente de objetos (Google Cloud Storage, con "
            "versionado habilitado) todo lo que no debe perderse, y tratar el disco local de la "
            "instancia como un caché regenerable. Tres categorías de artefactos viajan a GCS:")
    _bullet(doc, "Código versionado: el repositorio git se clona/pull en cada corrida; el código "
                 "siempre se recupera desde el control de versiones.")
    _bullet(doc, "Datos: los EDF crudos se guardan en un bucket. Solo los pacientes de train + "
                 "val se copian al disco local para entrenar (lectura rápida); los de test se "
                 "traen únicamente para la evaluación final. Los artefactos livianos "
                 "(split.json, scaler_stats.npz) se suben una vez y se leen desde "
                 "GCS/Drive.")
    _bullet(doc, "Artefactos de entrenamiento: checkpoints e historiales se "
                 "escriben localmente y se respaldan de forma incremental (ver 10.2).")
    doc.add_heading("10.2 Backups incrementales de checkpoints", level=2)
    _p(doc, "El entrenamiento incorpora el parámetro --backup-dir: un directorio espejo, "
            "apuntando a un bucket de GCS montado con gcsfuse (o a Google Drive montado en "
            "Colab), al que se copia el checkpoint y el historial cada vez que el criterio "
            "clínico mejora. Dos propiedades clave de este diseño:")
    _bullet(doc, "Copia en el momento de mejora, no al final: si la instancia se interrumpe a "
                 "mitad de corrida, el mejor modelo hasta ese instante ya está a salvo en GCS "
                 "junto con su historial completo por época (trazabilidad total de la curva de "
                 "aprendizaje).")
    _bullet(doc, "Tolerancia a fallos del respaldo: si la escritura al bucket falla (montaje "
                 "desmontado, cuota agotada), el entrenamiento continúa y el fallo solo se "
                 "registra como advertencia; nunca una falla de backup mata una corrida.")
    _p(doc, "La convención de nombres por escenario (models/<config>.pt) "
            "garantiza que los respaldos de diferentes configuraciones no se pisen, habilitando "
            "la comparación posterior y el versionado de artefactos.")
    doc.add_heading("10.3 Recuperación ante interrupciones y preempción", level=2)
    _p(doc, "Las instancias preemptibles y las sesiones de Colab pueden terminarse en cualquier "
            "momento. El protocolo de reanudación es: (1) re-clonar el código y re-montar los "
            "buckets; (2) comprobar qué escenarios ya tienen checkpoint e historial en GCS; "
            "(3) reanudar solo lo pendiente con la misma semilla, ratio, pos_weight y rutas, de "
            "modo que los resultados sean comparables con los ya existentes. El historial por "
            "época permite verificar en qué estado quedó cada corrida.")
    doc.add_heading("10.4 Reproducibilidad y auditoría", level=2)
    _p(doc, "Cada corrida persiste en su historial y checkpoint la semilla, ratio, pos_weight, "
            "learning rate, weight decay, batch size, desvío del ruido gaussiano (noise_std), "
            "versión de formato y punto de operación elegido. El selector verifica que el "
            "escenario haya guardado checkpoint antes de declararlo elegible, y el versionado de "
            "objetos en GCS permite recuperar cualquier estado anterior del entrenamiento.")
    doc.add_heading("10.5 Consideraciones de costo e I/O", level=2)
    _p(doc, "El pipeline es data-bound: el preprocesado (~1 s por archivo de hora) domina el "
            "tiempo de época frente al cómputo de la CNN. Por eso la estrategia prioriza "
            "mantener los EDF de desarrollo en SSD local/efímero y limitar las lecturas desde "
            "GCS a artefactos livianos. Si en el futuro se compara contra una CNN-2D, las "
            "métricas de costo (parámetros, FLOPs/ventana, latencia, VRAM) se reportarán por "
            "separado de las clínicas y normalizadas por ventana, siempre sobre el mismo "
            "hardware y con warm-up.")

    # ---------- 11. Criterios de éxito y resultados --------------------------
    doc.add_heading("11. Criterios de éxito", level=1)
    _p(doc, "El sistema se considera exitoso si, sobre la validación y luego sobre el "
            "test inter-paciente, cumple: (a) una tasa de falsas alarmas de evento por hora que "
            "respeta el techo clínico (≤ 10/h), de modo que el sistema sea tolerable para el clínico; "
            "(b) la mayor sensibilidad de evento posible entre los modelos que cumplen (a), conforme "
            "al criterio de selección; (c) una latencia de detección acotada (media y mediana "
            "reportadas). Las métricas de ventana (sensibilidad, especificidad, FPR, F1, MCC) se "
            "reportan como contexto, no como criterio de selección.")

    # ---------- 12. Limitaciones y trabajo futuro ----------------------------
    doc.add_heading("12. Limitaciones y trabajo futuro", level=1)
    _bullet(doc, "El sistema trabaja offline sobre EDF pregrabados; el paso a streaming en "
                 "tiempo real requiere un filtrado causal o por tramos (hoy es de fase cero) y "
                 "medir la latencia que eso reintroduce.")
    _bullet(doc, "La sensibilidad a nivel de evento depende del número de crisis por paciente; "
                 "con pocas crisis, un único falso negativo mueve mucho la métrica. Se debe "
                 "reportar el número de crisis y detecciones por conjunto (validación y test) "
                 "para valorar la robustez.")
    _bullet(doc, "El transfer learning sobre TUEV, la CNN-2D sobre espectrogramas y la "
                 "comparación de costo computacional quedan documentados como planes "
                 "independientes (PLAN_TRANSFER_LEARNING, PLAN_COMPUTO).")
    _bullet(doc, "Se planifican métricas de ranking (AUPRC) y reporte por paciente en versiones "
                 "futuras de la evaluación.")

    # ---------- 13. Conclusiones ---------------------------------------------
    doc.add_heading("13. Conclusiones", level=1)
    _p(doc, "Este trabajo define un flujo completo, reproducible y defendible para la detección "
            "de crisis epilépticas en EEG: preprocesamiento anti-fuga de datos, split "
            "inter-paciente con validación fija, selección de modelo por criterio clínico a nivel "
            "de evento, protocolo de experimentación con test reservado, y estrategia operativa "
             "en la nube con checkpoints respaldados de forma incremental. La decisión central — "
             "seleccionar el modelo maximizando la sensibilidad de evento sujeto a un techo de falsas alarmas por hora — "
             "traduce el requisito clínico a criterio objetivo de optimización y diferencia a "
             "este trabajo de los que evalúan solo a nivel de segmento.")

    # ---------- Referencias ---------------------------------------------------
    doc.add_heading("Referencias", level=1)
    refs = [
        "Wang X. et al., EEG-Based Seizure Onset Detection of Frontal and Temporal Lobe Epilepsies Using 1DCNN. IEEE TNSRE, vol. 33, 2025.",
        "Fawcett T., An Introduction to ROC Analysis. Pattern Recognition Letters 27(8), 2006.",
        "Wang et al., Scientific Reports, 2023 (s41598-023-41537-z).",
        "Kashefi Amiri et al., Scientific Reports, 2025 (s41598-025-18479-9).",
        "Saha, Tchinda et al., Heliyon, 2025 (PIIS240584402501374X).",
        "He H., García E. A., Learning from Imbalanced Data. IEEE TKDE 21(9), 2009.",
        "Branco P., Torgo L., Ribeiro R. P., A Survey of Predictive Modeling on Imbalanced Domains. ACM Computing Surveys 49(2), 2016.",
        "LeCun Y. et al., Efficient BackProp. En Neural Networks: Tricks of the Trade, 1998.",
        "Ioffe S., Szegedy C., Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv:1502.03167, 2015.",
        "Schirrmeister R. T. et al., Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization. Human Brain Mapping, 2017.",
        "Roy Y. et al., Deep Learning-Based Electroencephalography Analysis: A Systematic Review. arXiv:1901.05498, 2019.",
        "CHB-MIT Scalp EEG Database, PhysioNet. https://physionet.org/content/chbmit/1.0.0/",
    ]
    for r in refs:
        _bullet(doc, r)

    doc.add_page_break()
    # ---------- Glosario ------------------------------------------------------
    doc.add_heading("Glosario", level=1)
    glosario = [
        ("Evento (crisis)", "Crisis completa anotada; unidad de evaluación clínica."),
        ("Segmento / ventana", "Unidad de clasificación (5.12 s de señal)."),
        ("Falsas alarmas de evento por hora", "Falsas alarmas divididas por las horas de registro cubiertas."),
        ("Ruido gaussiano (augmentation)", "Ruido aditivo N(0, σ) sobre las ventanas de crisis durante el train para diversificar la clase minoritaria."),
        ("Leakage", "Fuga de información de test hacia el entrenamiento que invalida las métricas."),
        ("pos_weight", "Peso de la clase positiva en la BCE ponderada."),
        ("Undersampling", "Submuestreo de la clase mayoritaria para balancear cada época."),
        ("Punto de operación", "Umbral τ más sus métricas resultantes (sensibilidad, falsas alarmas por hora)."),
        ("Latencia", "Tiempo desde el onset de la crisis hasta la primera alarma."),
    ]
    _table(doc, ["Término", "Definición"], glosario, widths=[5.0, 11.0])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_FILE))
    return OUT_FILE


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = build()
    print(f"Informe generado: {out}")
