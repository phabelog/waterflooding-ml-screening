"""
Preprocesamiento del conjunto de datos.

Corresponde a la Etapa 3 del flujo de trabajo metodológico:

    3.1  Limpieza y tratamiento de datos faltantes y valores atípicos
    3.2  División estratificada en entrenamiento y prueba
    3.3  Escalado y selección de características
    3.4  SMOTE solo en entrenamiento, si es necesario

Principio que ordena todo el módulo: NINGUNA transformación aprende nada del
conjunto de prueba. Por eso el escalado no se aplica aquí sobre los datos,
sino que se entrega como un transformador SIN AJUSTAR, que la Etapa 4 coloca
dentro de un Pipeline. Así se ajusta solo con los datos de entrenamiento de
cada partición de la validación cruzada, y el conjunto de prueba permanece
intacto hasta la evaluación final de la Etapa 5.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "dataset_generation"))
sys.path.insert(0, os.path.join(_HERE, "..", "analysis"))

from generate_dataset import FEATURE_COLUMNS  # noqa: E402
from data_analysis import class_balance       # noqa: E402

TARGET = "label"

# Semilla única del proyecto para la división y la validación cruzada.
RANDOM_STATE = 42

# Proporción del conjunto de prueba (retenido hasta la Etapa 5).
TEST_SIZE = 0.20

# Número de particiones de la validación cruzada estratificada (Etapa 4).
CV_FOLDS = 5

# Variables con asimetría elevada (Etapa 2: k 1.89, muo 2.78). Se
# transforman con log10 antes de estandarizar.
LOG_FEATURES = ["k_md", "muo_cp"]


# ======================================================================
#  3.1  LIMPIEZA: DATOS FALTANTES Y VALORES ATÍPICOS
# ======================================================================

def missing_and_duplicates(df, columns=None):
    """Datos faltantes por variable y filas duplicadas en las entradas."""
    cols = columns or FEATURE_COLUMNS
    faltantes = df[cols].isna().sum()
    return {
        "faltantes_por_variable": faltantes[faltantes > 0].to_dict(),
        "faltantes_total": int(faltantes.sum()),
        "duplicados": int(df[cols].duplicated().sum()),
    }


def iqr_outliers(df, columns=None, log_columns=None, k=1.5):
    """
    Valores atípicos según el criterio intercuartílico (Tukey, k = 1.5).

    Con log_columns se evalúan esas variables en escala log10, que es la
    escala en que el modelo las recibirá.
    """
    cols = columns or FEATURE_COLUMNS
    log_columns = log_columns or []
    conteo = {}
    for c in cols:
        x = np.log10(df[c]) if c in log_columns else df[c]
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        n = int(((x < q1 - k * iqr) | (x > q3 + k * iqr)).sum())
        if n:
            conteo[c] = n
    return conteo


def domain_report(df_new, df_reference, columns=None):
    """
    Revisa un conjunto nuevo (por ejemplo, los datos de EP Petroecuador)
    contra el dominio del conjunto de entrenamiento.

    Reporta, por variable:
      - faltantes : valores ausentes ("N/D" en la solicitud de datos)
      - fuera_rango : valores fuera del mínimo-máximo del entrenamiento

    Política adoptada: los registros con variables faltantes NO se imputan.
    Imputar introduciría supuestos del propio estudio en un conjunto cuyo
    propósito es precisamente evaluar el modelo a ciegas. Los registros
    fuera de rango se conservan pero se señalan, porque en ellos el modelo
    extrapola y su predicción debe leerse con cautela.
    """
    cols = columns or FEATURE_COLUMNS
    filas = []
    for c in cols:
        if c not in df_new.columns:
            filas.append({"variable": c, "faltantes": len(df_new),
                          "fuera_rango": 0, "nota": "columna ausente"})
            continue
        lo, hi = df_reference[c].min(), df_reference[c].max()
        v = pd.to_numeric(df_new[c], errors="coerce")
        filas.append({"variable": c,
                      "faltantes": int(v.isna().sum()),
                      "fuera_rango": int(((v < lo) | (v > hi)).sum()),
                      "nota": ""})
    rep = pd.DataFrame(filas).set_index("variable")
    completos = df_new.reindex(columns=cols).apply(
        pd.to_numeric, errors="coerce").notna().all(axis=1)
    return rep, int(completos.sum())


# ======================================================================
#  3.2  DIVISIÓN ESTRATIFICADA
# ======================================================================

def split_train_test(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    División estratificada en entrenamiento y prueba.

    La estratificación conserva en ambos conjuntos la misma proporción de
    casos aptos. El conjunto de prueba queda RETENIDO: no participa en el
    ajuste del escalado, en la validación cruzada ni en la optimización de
    hiperparámetros, y se utiliza una única vez en la Etapa 5.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=random_state)


def get_cv(n_splits=CV_FOLDS, random_state=RANDOM_STATE):
    """
    Validación cruzada estratificada que empleará la Etapa 4, definida aquí
    para que todos los modelos se comparen sobre las mismas particiones.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=random_state)


# ======================================================================
#  3.3  ESCALADO Y SELECCIÓN DE CARACTERÍSTICAS
# ======================================================================

def build_preprocessor(columns=None):
    """
    Transformador de escalado, entregado SIN AJUSTAR.

      - k_md y muo_cp : log10 y luego estandarización (media 0, desv. 1).
      - resto         : estandarización.

    La transformación logarítmica corrige la asimetría detectada en la
    Etapa 2 y es físicamente natural: la permeabilidad y la viscosidad
    actúan sobre el flujo de forma multiplicativa, a través de cocientes
    como la relación de movilidad.

    La estandarización es necesaria para la SVM, sensible a la escala de las
    variables. Para los modelos de árboles (árbol de decisión, Random
    Forest, XGBoost) es irrelevante, porque sus particiones no dependen de
    transformaciones monótonas; se aplica igualmente a todos los modelos
    para que la comparación de la Etapa 4 parta de las mismas entradas.
    """
    cols = columns or FEATURE_COLUMNS
    log_cols = [c for c in LOG_FEATURES if c in cols]
    otras = [c for c in cols if c not in log_cols]
    log_pipe = Pipeline([
        ("log10", FunctionTransformer(np.log10, feature_names_out="one-to-one")),
        ("escala", StandardScaler()),
    ])
    return ColumnTransformer(
        [("log", log_pipe, log_cols), ("std", StandardScaler(), otras)],
        verbose_feature_names_out=False,
    )


# Variables que no intervienen en ninguna ecuación del framework ni en los
# criterios de cribado. Se CONSERVAN como entradas a modo de controles
# negativos (ver feature_selection_summary).
NEGATIVE_CONTROLS = ["phi", "depth_ft"]


def feature_selection_summary():
    """
    Decisión sobre la selección de características, con su fundamento.

    Se conservan las 16 variables de entrada. La redundancia ya se resolvió
    en la Etapa 2 (exclusión de la temperatura). No se aplica una selección
    estadística adicional porque eliminaría variables que sí intervienen en
    la física pero cuyo efecto marginal es débil en el rango muestreado
    (por ejemplo Sor o los exponentes de Corey): débil no es irrelevante, y
    esa distinción corresponde al análisis de importancia de la Etapa 5.

    La porosidad y la profundidad no intervienen en el cálculo del
    framework: la porosidad determina el volumen de petróleo en sitio pero
    no la fracción recuperable, y la profundidad no entra en ninguna
    ecuación ni criterio. Se mantienen como CONTROLES NEGATIVOS: si el
    modelo les asignara importancia, revelaría que aprende ruido del
    muestreo y no la física. Además, forman parte de los criterios del
    cribado tradicional, con el que el modelo se compara en la Etapa 5.
    """
    return {
        "variables": list(FEATURE_COLUMNS),
        "n_variables": len(FEATURE_COLUMNS),
        "transformacion_log10": list(LOG_FEATURES),
        "controles_negativos": list(NEGATIVE_CONTROLS),
    }


# ======================================================================
#  3.4  TRATAMIENTO DEL DESBALANCE
# ======================================================================

def resampling_decision(y_train):
    """
    Decide si corresponde aplicar SMOTE, con los MISMOS umbrales empleados
    en el diagnóstico de la Etapa 2 (data_analysis.class_balance).

      clase minoritaria >= 25 %  -> ponderación de clases, sin SMOTE
      clase minoritaria <  25 %  -> SMOTE, solo sobre el entrenamiento

    Devuelve además los pesos que emplearán los modelos de la Etapa 4:
      - class_weight = "balanced"  (árbol de decisión, Random Forest, SVM)
      - scale_pos_weight = n_no_aptos / n_aptos  (XGBoost)
    """
    diag = class_balance(pd.DataFrame({TARGET: np.asarray(y_train)}))
    usar_smote = diag["clase_minoritaria"] < 0.25
    n_pos = diag["n_apto"]
    n_neg = diag["n_no_apto"]
    return {
        "clase_minoritaria": diag["clase_minoritaria"],
        "diagnostico": diag["diagnostico"],
        "usar_smote": bool(usar_smote),
        "class_weight": "balanced",
        "scale_pos_weight": (n_neg / n_pos) if n_pos else 1.0,
    }


# ======================================================================
#  RESUMEN
# ======================================================================

def run_preprocessing(df, verbose=True):
    """Ejecuta las cuatro partes de la Etapa 3 y devuelve sus resultados."""
    limpieza = missing_and_duplicates(df)
    atip_orig = iqr_outliers(df)
    atip_log = iqr_outliers(df, log_columns=LOG_FEATURES)
    X_train, X_test, y_train, y_test = split_train_test(df)
    desbalance = resampling_decision(y_train)

    out = {
        "limpieza": limpieza,
        "atipicos_escala_original": atip_orig,
        "atipicos_tras_log10": atip_log,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "seleccion": feature_selection_summary(),
        "desbalance": desbalance,
    }

    if verbose:
        print(f"Faltantes: {limpieza['faltantes_total']} | "
              f"duplicados: {limpieza['duplicados']}")
        print(f"Atípicos (IQR) en escala original: {atip_orig}")
        print(f"Atípicos (IQR) tras log10:         {atip_log or 'ninguno'}")
        print(f"Entrenamiento: {len(X_train)} ({y_train.mean()*100:.1f} % aptos) | "
              f"Prueba: {len(X_test)} ({y_test.mean()*100:.1f} % aptos)")
        print(f"Variables de entrada: {out['seleccion']['n_variables']}")
        print(f"Desbalance: {desbalance['diagnostico']} -> "
              f"SMOTE: {'sí' if desbalance['usar_smote'] else 'no'}; "
              f"scale_pos_weight = {desbalance['scale_pos_weight']:.3f}")
    return out
