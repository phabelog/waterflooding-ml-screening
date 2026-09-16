"""
Análisis del conjunto de datos sintético.

Corresponde a la Etapa 2 del flujo de trabajo metodológico:

    2.1  Estadísticas descriptivas y distribución de las variables
    2.2  Análisis de correlación
    2.3  Evaluación de balance de clases
    2.4  Verificación de consistencia física del dataset

El módulo no modifica los datos: solo los describe y los audita. Cualquier
transformación (limpieza, escalado, división en entrenamiento y prueba)
corresponde a la Etapa 3 y se implementa aparte.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dataset_generation"))

from generate_dataset import FEATURE_COLUMNS, DERIVED_COLUMNS


# ======================================================================
#  2.1  ESTADÍSTICAS DESCRIPTIVAS
# ======================================================================

def descriptive_statistics(df, columns=None):
    """
    Tabla de estadísticas descriptivas: conteo, media, desviación estándar,
    mínimo, cuartiles, máximo, asimetría y curtosis.
    """
    cols = columns or FEATURE_COLUMNS
    stats = df[cols].describe().T
    stats["asimetría"] = df[cols].skew()
    stats["curtosis"] = df[cols].kurtosis()
    return stats.rename(columns={"count": "n", "mean": "media",
                                 "std": "desv_est", "min": "mín", "max": "máx"})


def distribution_by_class(df, columns=None):
    """
    Media de cada variable separada por clase, con la diferencia
    estandarizada de Cohen como medida del poder discriminante individual.
    """
    cols = columns or FEATURE_COLUMNS
    apto, no_apto = df[df["label"] == 1], df[df["label"] == 0]

    rows = []
    for c in cols:
        m1, m0 = apto[c].mean(), no_apto[c].mean()
        s1, s0 = apto[c].std(), no_apto[c].std()
        n1, n0 = len(apto), len(no_apto)
        s_pool = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
        d = (m1 - m0) / s_pool if s_pool > 0 else 0.0
        rows.append({"variable": c, "media_apto": m1,
                     "media_no_apto": m0, "d_de_Cohen": d})

    out = pd.DataFrame(rows).set_index("variable")
    return out.reindex(out["d_de_Cohen"].abs().sort_values(ascending=False).index)


# ======================================================================
#  2.2  CORRELACIÓN
# ======================================================================

def correlation_matrix(df, columns=None, method="spearman"):
    """
    Matriz de correlación entre variables de entrada.

    Se usa Spearman por defecto: varias relaciones del framework son
    monótonas pero no lineales (por ejemplo, permeabilidad e inyectividad),
    y Pearson las subestimaría.
    """
    cols = columns or FEATURE_COLUMNS
    return df[cols].corr(method=method)


def redundant_pairs(df, columns=None, threshold=0.85, method="spearman"):
    """
    Pares de variables con correlación por encima del umbral.

    Una correlación muy alta entre dos entradas indica información
    duplicada: ambas aportan lo mismo y el modelo repartirá la importancia
    entre ellas de forma arbitraria, lo que dificulta la interpretación.
    """
    corr = correlation_matrix(df, columns, method)
    cols = corr.columns
    pares = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pares.append({"var_1": cols[i], "var_2": cols[j], "r": r})
    return pd.DataFrame(pares)


def correlation_with_label(df, columns=None):
    """
    Correlación punto-biserial de cada variable con la etiqueta.

    Detecta dos situaciones opuestas: una variable que por sí sola determine
    la etiqueta (criterio de cribado demasiado simple) o un conjunto sin
    ninguna señal (problema no aprendible).
    """
    cols = columns or FEATURE_COLUMNS
    r = {c: np.corrcoef(df[c], df["label"])[0, 1] for c in cols}
    out = pd.DataFrame({"r_con_etiqueta": pd.Series(r)})
    return out.reindex(out["r_con_etiqueta"].abs()
                       .sort_values(ascending=False).index)


# ======================================================================
#  2.3  BALANCE DE CLASES
# ======================================================================

def class_balance(df):
    """Conteo y proporción de cada clase, con el diagnóstico asociado."""
    n = len(df)
    n_apto = int((df["label"] == 1).sum())
    prop = n_apto / n if n else 0.0
    minoritaria = min(prop, 1 - prop)

    if minoritaria >= 0.40:
        diag = "Balanceado: no requiere remuestreo"
    elif minoritaria >= 0.25:
        diag = "Desbalance leve: manejable con ponderación de clases"
    elif minoritaria >= 0.10:
        diag = "Desbalance moderado: considerar SMOTE en entrenamiento"
    else:
        diag = "Desbalance severo: SMOTE o remuestreo necesario"

    return {"n_total": n, "n_apto": n_apto, "n_no_apto": n - n_apto,
            "proporcion_apto": prop, "clase_minoritaria": minoritaria,
            "diagnostico": diag}


def rejection_breakdown(df):
    """
    Frecuencia de cada criterio de rechazo entre los casos no aptos.

    Si un único criterio concentrara casi todos los rechazos, el problema se
    reduciría a ese umbral y el modelo no aprendería una frontera
    multivariable.
    """
    from collections import Counter
    conteo = Counter()
    for texto in df.loc[df["label"] == 0, "reject_reasons"].fillna(""):
        for parte in [p for p in str(texto).split("; ") if p]:
            conteo[parte.split("=")[0].strip()] += 1

    n_no = int((df["label"] == 0).sum())
    out = pd.DataFrame({"criterio": list(conteo.keys()),
                        "casos": list(conteo.values())})
    out["% de no aptos"] = out["casos"] / n_no * 100
    return out.sort_values("casos", ascending=False).reset_index(drop=True)


# ======================================================================
#  2.4  CONSISTENCIA FÍSICA
# ======================================================================

def physical_consistency(df):
    """
    Auditoría de consistencia física del dataset ya generado.

    Es una verificación independiente de la de la Etapa 1: allí se comprueba
    que el framework calcula bien; aquí, que el archivo resultante no
    contiene valores imposibles.
    """
    chequeos = []

    def add(nombre, mask_valido):
        n_fallo = int((~mask_valido).sum())
        chequeos.append({"verificación": nombre, "casos_inválidos": n_fallo,
                         "resultado": "PASA" if n_fallo == 0 else "FALLA"})

    add("Saturación móvil positiva (1 - Swi - Sor > 0)",
        (1 - df["Swi"] - df["Sor"]) > 0)
    add("So actual entre Sor y 1 - Swi",
        (df["So_current"] >= df["Sor"] - 1e-9) &
        (df["So_current"] <= 1 - df["Swi"] + 1e-9))
    add("Saturaciones dentro del rango [0, 1]",
        df[["Swi", "Sor", "So_current"]].apply(
            lambda s: s.between(0, 1)).all(axis=1))
    add("Eficiencias de barrido en (0, 1]",
        df["EA"].between(0, 1, inclusive="right") &
        df["EV"].between(0, 1, inclusive="right"))
    add("Eficiencia de desplazamiento en [0, 1]", df["ED"].between(0, 1))
    add("RF_total = ED x EA x EV",
        (df["RF_total"] - df["ED"] * df["EA"] * df["EV"]).abs() < 1e-6)
    add("RF_total nunca supera ED", df["RF_total"] <= df["ED"] + 1e-9)
    add("Relacion de movilidad positiva", df["M"] > 0)
    add("Caudal de inyeccion positivo", df["q_inj_bpd"] > 0)
    add("Permeabilidad y porosidad positivas",
        (df["k_md"] > 0) & (df["phi"] > 0))
    add("Sin valores infinitos",
        np.isfinite(df.select_dtypes(include=[np.number])).all(axis=1))

    faltantes = int(df[FEATURE_COLUMNS + DERIVED_COLUMNS].isna().sum().sum())
    chequeos.append({"verificación": "Sin datos faltantes en variables y derivadas",
                     "casos_inválidos": faltantes,
                     "resultado": "PASA" if faltantes == 0 else "FALLA"})

    return pd.DataFrame(chequeos)


def monotonicity_checks(df, n_bins=10):
    """
    Comprueba que las tendencias esperadas por la fisica se observan en el
    dataset agregado:

      - a mayor relacion de movilidad, menor eficiencia de barrido areal;
      - a mayor heterogeneidad, menor eficiencia de barrido vertical;
      - a mayor recobro total, mayor proporcion de casos aptos.
    """
    out = {}
    b = pd.qcut(df["M"], n_bins, duplicates="drop")
    out["EA_vs_M"] = df.groupby(b, observed=True)["EA"].mean()
    b = pd.qcut(df["Vdp"], n_bins, duplicates="drop")
    out["EV_vs_Vdp"] = df.groupby(b, observed=True)["EV"].mean()
    b = pd.qcut(df["RF_total"], n_bins, duplicates="drop")
    out["apto_vs_RF"] = df.groupby(b, observed=True)["label"].mean()
    return out


# ======================================================================
#  RESUMEN
# ======================================================================

def run_full_analysis(df, verbose=True):
    """Ejecuta las cuatro partes de la Etapa 2 y devuelve los resultados."""
    resultados = {
        "descriptivas": descriptive_statistics(df),
        "por_clase": distribution_by_class(df),
        "correlacion": correlation_matrix(df),
        "redundantes": redundant_pairs(df),
        "corr_etiqueta": correlation_with_label(df),
        "balance": class_balance(df),
        "rechazos": rejection_breakdown(df),
        "consistencia": physical_consistency(df),
        "monotonia": monotonicity_checks(df),
    }

    if verbose:
        b = resultados["balance"]
        print(f"Escenarios: {b['n_total']}  |  aptos {b['n_apto']} "
              f"({b['proporcion_apto']*100:.1f} %)  |  no aptos {b['n_no_apto']}")
        print(f"Balance: {b['diagnostico']}")
        cons = resultados["consistencia"]
        n_ok = int((cons["resultado"] == "PASA").sum())
        print(f"Consistencia fisica: {n_ok}/{len(cons)} verificaciones superadas")
        red = resultados["redundantes"]
        if len(red):
            print(f"Pares redundantes (|r| >= 0.85): {len(red)}")
            for _, r in red.iterrows():
                print(f"   {r['var_1']} - {r['var_2']}: r = {r['r']:+.3f}")
        else:
            print("Sin pares de variables redundantes")

    return resultados
