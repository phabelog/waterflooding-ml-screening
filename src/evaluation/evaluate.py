"""
Evaluación y validación de los modelos.

Corresponde a la Etapa 5 del flujo de trabajo metodológico:

    5.1  Evaluación única sobre el conjunto de prueba (1 000 escenarios)
    5.2  Comparación entrenamiento / validación cruzada / prueba
    5.3  Métricas: matriz de confusión, precisión, recall, F1, exactitud
         y área bajo la curva ROC (AUC-ROC)
    5.4  Importancia de variables por permutación y controles negativos
    5.5  Comparación con el cribado tradicional por rangos
    5.6  Caso de referencia con datos de campo (contexto, no validación)

Reglas que este módulo respeta:
  - Los hiperparámetros son los de la Etapa 4 (results/etapa4_resultados.json);
    aquí no se ajusta nada con el conjunto de prueba.
  - El modelo final se elige por su F1 en VALIDACIÓN CRUZADA, antes de mirar
    la prueba. La prueba solo informa el desempeño esperado.
  - Los datos de campo no entrenan ni validan el modelo: sirven para
    comprobar que el dominio sintético cubre condiciones reales y para
    ilustrar el uso de la herramienta.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import cross_validate

_HERE = os.path.dirname(__file__)
for _sub in ("models", "preprocessing", "analysis", "dataset_generation", "physics"):
    sys.path.insert(0, os.path.join(_HERE, "..", _sub))

import preprocess as pp                                   # noqa: E402
import train_models as tm                                 # noqa: E402
from generate_dataset import FEATURE_COLUMNS, compute_physics   # noqa: E402
from labeling import THRESHOLDS, evaluate_scenario        # noqa: E402
from variable_ranges import (INPUT_VARIABLES, RELPERM_VARIABLES,   # noqa: E402
                             oil_viscosity_from_api,
                             water_viscosity_from_temperature)

RANDOM_STATE = pp.RANDOM_STATE
METRICAS = ["f1", "precision", "recall", "accuracy", "auc_roc"]

NOMBRES_VARIABLES = {
    "k_md": "Permeabilidad k", "phi": "Porosidad φ", "h_net_ft": "Espesor neto h",
    "api": "Gravedad API", "Swi": "Swi", "Sor": "Sor", "Vdp": "Dykstra-Parsons Vdp",
    "depth_ft": "Profundidad D", "skin": "Skin", "kro_max": "kro máx.",
    "krw_max": "krw máx.", "no": "Exponente no", "nw": "Exponente nw",
    "muo_cp": "Viscosidad del petróleo μo", "muw_cp": "Viscosidad del agua μw",
    "So_current": "Saturación actual So",
}


# ----------------------------------------------------------------------
#  Utilidades
# ----------------------------------------------------------------------

def continuous_score(model, X):
    """
    Puntaje continuo de la clase apta, para la curva ROC.

    Los modelos de árboles entregan probabilidades. La SVM se entrenó sin
    estimación de probabilidades (Etapa 4), así que se usa su función de
    decisión: la curva ROC solo necesita un puntaje que ordene los casos.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, score=None):
    out = {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    out["auc_roc"] = float(roc_auc_score(y_true, score)) if score is not None else None
    return out


def select_final_model(resultados_e4):
    """Modelo con mayor F1 medio en validación cruzada (Etapa 4)."""
    cv = {n: d["validacion_cruzada"]["f1_media"]
          for n, d in resultados_e4["modelos"].items()}
    return max(cv, key=cv.get)


# ----------------------------------------------------------------------
#  5.1 - 5.3  Ajuste final y métricas en entrenamiento, validación y prueba
# ----------------------------------------------------------------------

def fit_final_models(df, resultados_e4, models=None):
    """
    Reconstruye cada modelo con los hiperparámetros de la Etapa 4 y lo
    ajusta con TODO el entrenamiento (4 000 escenarios).
    """
    X_train, X_test, y_train, y_test = pp.split_train_test(df)
    spw = pp.resampling_decision(y_train)["scale_pos_weight"]
    models = models or [m for m in tm.MODEL_NAMES if m in resultados_e4["modelos"]]
    fitted = {}
    for name in models:
        if name == "xgboost" and not tm.xgboost_available():
            continue
        pipe = tm.build_best_model(
            name, resultados_e4["modelos"][name]["mejores_hiperparametros"], spw)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    return fitted, (X_train, X_test, y_train, y_test)


def cv_auc(model, X_train, y_train, n_jobs=1):
    """
    AUC-ROC en validación cruzada con las mismas 5 particiones de la Etapa 3
    y los hiperparámetros ya fijados (no se vuelve a buscar).
    """
    r = cross_validate(clone(model), X_train, y_train, cv=pp.get_cv(),
                       scoring="roc_auc", n_jobs=n_jobs)
    return float(r["test_score"].mean()), float(r["test_score"].std())


def evaluate_models(fitted, split, resultados_e4, n_jobs=1):
    """
    Tabla de métricas por modelo en tres conjuntos:
      - entrenamiento : el modelo ajustado con los 4 000, evaluado en ellos
      - validacion    : media de las 5 particiones (Etapa 4; AUC calculada aquí)
      - prueba        : evaluación única sobre los 1 000 escenarios apartados
    """
    X_train, X_test, y_train, y_test = split
    filas, extras = [], {}
    for name, model in fitted.items():
        # entrenamiento
        m_tr = compute_metrics(y_train, model.predict(X_train),
                               continuous_score(model, X_train))
        # validación cruzada
        cv = resultados_e4["modelos"][name]["validacion_cruzada"]
        auc_m, auc_s = cv_auc(model, X_train, y_train, n_jobs)
        m_cv = {"f1": cv["f1_media"], "precision": cv["precision_media"],
                "recall": cv["recall_media"], "accuracy": cv["accuracy_media"],
                "auc_roc": auc_m}
        # prueba
        y_pred = model.predict(X_test)
        s_te = continuous_score(model, X_test)
        m_te = compute_metrics(y_test, y_pred, s_te)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr, tpr, _ = roc_curve(y_test, s_te)

        for conj, m in (("entrenamiento", m_tr), ("validacion", m_cv), ("prueba", m_te)):
            filas.append({"modelo": name, "conjunto": conj, **m})
        extras[name] = {
            "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "roc": {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr]},
            "f1_cv_desv": cv["f1_desv"], "auc_cv_desv": auc_s,
        }
    tabla = pd.DataFrame(filas).set_index(["modelo", "conjunto"])
    return tabla, extras


def trivial_baseline(y_train, y_test):
    """Clasificador trivial: declarar apto a todos los casos."""
    y_pred = np.ones_like(np.asarray(y_test))
    m = compute_metrics(y_test, y_pred)
    m["auc_roc"] = 0.5
    return m


# ----------------------------------------------------------------------
#  5.4  Importancia de variables
# ----------------------------------------------------------------------

def variable_importance(model, X_test, y_test, n_repeats=30):
    """
    Importancia por permutación sobre la prueba: cuánto cae el F1 al
    desordenar una variable. No depende del tipo de modelo.

    Los controles negativos (porosidad y profundidad) no intervienen en el
    framework físico: su importancia debe ser indistinguible de cero.
    """
    r = permutation_importance(model, X_test, y_test, scoring="f1",
                               n_repeats=n_repeats, random_state=RANDOM_STATE)
    imp = pd.DataFrame({"importancia": r.importances_mean,
                        "desv": r.importances_std}, index=X_test.columns)
    imp["control_negativo"] = imp.index.isin(pp.NEGATIVE_CONTROLS)
    return imp.sort_values("importancia", ascending=False)


# ----------------------------------------------------------------------
#  5.5  Cribado tradicional por rangos
# ----------------------------------------------------------------------

def traditional_screening(X, th=None):
    """
    Cribado tradicional: cada propiedad se compara por separado con su
    umbral y el reservorio pasa si todas cumplen.

    Usa solo los criterios que se pueden leer directamente de las
    propiedades de entrada (k, h, So, μo y Vdp). No calcula la relación de
    movilidad, el recobro ni el caudal de inyección, que dependen de la
    combinación de varias propiedades: esa es la diferencia con el
    etiquetado físico y con el modelo de ML.
    """
    th = th or THRESHOLDS
    ok = ((X["k_md"] >= th["k_min_md"]) & (X["h_net_ft"] >= th["h_net_min_ft"])
          & (X["So_current"] >= th["So_min"]) & (X["muo_cp"] <= th["muo_max_cp"])
          & (X["Vdp"] <= th["Vdp_max"]))
    return ok.astype(int).to_numpy()


# ----------------------------------------------------------------------
#  5.6  Caso de referencia con datos de campo (contexto)
# ----------------------------------------------------------------------
# Arena de un campo maduro del Activo Auca con inyección de agua en marcha
# (datos de EP Petroecuador, anonimizados). Valores promedio reportados:
REFERENCE_FIXED = {
    "k_md": 680.0,       # permeabilidad promedio de la zona de pago
    "phi": 0.16,         # porosidad efectiva promedio
    "h_net_ft": 43.0,    # espesor de pago promedio
    "api": 19.0,         # gravedad API (pruebas de producción: 18.5-19)
    "depth_ft": 9600.0,  # profundidad vertical del tope (9 464-9 740 ft)
}
# Propiedades con evidencia parcial o sin medición: se sortean dentro del
# intervalo indicado (y siempre dentro del dominio de entrenamiento).
REFERENCE_UNCERTAIN = {
    "Swi": (0.10, 0.16),       # registros (prom. 0.11) y núcleo (0.16)
    "Sor": (0.29, 0.35),       # puntos extremos de la arena; un núcleo dio 0.38
    "krw_max": (0.15, 0.25),   # curvas de la arena: krw final menor de 0.25
    "T_res_F": (210.0, 230.0), # sin medición directa; referencia: 226 °F
    # Sin dato de campo: todo el rango de entrenamiento
    "Vdp": INPUT_VARIABLES["Vdp"][:2],
    "skin": INPUT_VARIABLES["skin"][:2],
    "kro_max": RELPERM_VARIABLES["kro_max"][:2],
    "no": RELPERM_VARIABLES["no"][:2],
    "nw": RELPERM_VARIABLES["nw"][:2],
}


def reference_case_scenarios(n=2000, seed=RANDOM_STATE):
    """
    Genera n realizaciones del caso de referencia: las propiedades medidas
    quedan fijas y las inciertas se sortean en su intervalo.

    La saturación de petróleo al inicio de la inyección se aproxima a
    1 - Swi: antes de inyectar, los pozos producían con cortes de agua
    bajos y la producción primaria había recuperado menos del 10 % del
    petróleo, merma que el aumento del factor volumétrico al caer la
    presión compensa en parte.
    """
    rng = np.random.default_rng(seed)
    d = {k: np.full(n, v) for k, v in REFERENCE_FIXED.items()}
    for k, (lo, hi) in REFERENCE_UNCERTAIN.items():
        d[k] = rng.uniform(lo, hi, n)
    d["muo_cp"] = oil_viscosity_from_api(d["api"], rng)
    d["muw_cp"] = water_viscosity_from_temperature(d["T_res_F"])
    d["So_current"] = 1.0 - d["Swi"]
    return pd.DataFrame(d)


def reference_case(model, df_train_reference, n=2000, seed=RANDOM_STATE):
    """
    Aplica el framework físico (etiqueta) y el modelo de ML a las
    realizaciones del caso de referencia.
    """
    esc = reference_case_scenarios(n, seed)
    fis = []
    for _, row in esc.iterrows():
        ph = compute_physics(row)
        ev = evaluate_scenario(row.to_dict(), ph)
        fis.append({**ph, "label_fisica": ev["label"],
                    "motivos": "; ".join(ev["reject_reasons"])})
    fis = pd.DataFrame(fis)
    X = esc[FEATURE_COLUMNS]
    pred = model.predict(X)
    prob = continuous_score(model, X)
    # Dominio: se cuentan las realizaciones con alguna variable fuera del
    # mínimo-máximo del entrenamiento por más del 1 % del rango (los valores
    # sorteados en el borde exacto del rango no cuentan como extrapolación).
    ref = df_train_reference[FEATURE_COLUMNS]
    tol = 0.01 * (ref.max() - ref.min())
    fuera = ((X < ref.min() - tol) | (X > ref.max() + tol)).any(axis=1)
    out = pd.concat([esc, fis], axis=1)
    out["pred_modelo"] = pred
    out["puntaje_modelo"] = prob
    resumen = {
        "realizaciones": int(n),
        "fraccion_apta_fisica": float(fis["label_fisica"].mean()),
        "fraccion_apta_modelo": float(pred.mean()),
        "coincidencia_modelo_fisica": float((pred == fis["label_fisica"]).mean()),
        "probabilidad_media_modelo": float(np.mean(prob)),
        "M_p10_p50_p90": [float(v) for v in np.percentile(fis["M"], [10, 50, 90])],
        "RF_p10_p50_p90": [float(v) for v in np.percentile(fis["RF_total"], [10, 50, 90])],
        "q_iny_p10_p50_p90": [float(v) for v in np.percentile(fis["q_inj_bpd"], [10, 50, 90])],
        "muo_p10_p50_p90": [float(v) for v in np.percentile(esc["muo_cp"], [10, 50, 90])],
        "realizaciones_fuera_de_dominio": int(fuera.sum()),
        "motivos_rechazo_fisica": {
            k: int(v) for k, v in pd.Series(
                [m.split("=")[0].strip() for ms in fis["motivos"] if ms for m in ms.split("; ")]
            ).value_counts().items()},
    }
    return out, resumen


# Rangos de entrenamiento frente a los valores de campo (para el artículo)
REFERENCE_TABLE = [
    # variable, unidad, rango sintético, valor de campo, dentro del rango
    ("Permeabilidad", "mD", "5 – 3 000", "680 (217 – 1 180)", "Sí"),
    ("Porosidad", "fracción", "0.08 – 0.25", "0.16 (0.13 – 0.19)", "Sí"),
    ("Espesor neto", "ft", "5 – 80", "43 (28 – 56)", "Sí"),
    ("Gravedad API", "°API", "14 – 35", "18.5 – 19", "Sí"),
    ("Profundidad vertical", "ft", "7 000 – 10 500", "9 464 – 9 740", "Sí"),
    ("Saturación de agua irreducible", "fracción", "0.10 – 0.45", "0.06 – 0.21", "Parcial"),
    ("Saturación residual de petróleo", "fracción", "0.15 – 0.35", "0.29 – 0.38", "Parcial"),
    ("krw máximo", "fracción", "0.15 – 0.55", "< 0.25", "Sí"),
]


# ----------------------------------------------------------------------
#  Ejecución completa
# ----------------------------------------------------------------------

def run_evaluation(df, resultados_e4, n_jobs=1, verbose=True):
    """Ejecuta toda la Etapa 5 y devuelve un diccionario serializable."""
    fitted, split = fit_final_models(df, resultados_e4)
    X_train, X_test, y_train, y_test = split
    if verbose:
        print("Modelos ajustados con el entrenamiento:", ", ".join(fitted))

    tabla, extras = evaluate_models(fitted, split, resultados_e4, n_jobs)
    final = select_final_model({"modelos": {k: v for k, v in resultados_e4["modelos"].items()
                                            if k in fitted}})
    if verbose:
        print("Modelo final (mayor F1 en validación cruzada):", tm.MODEL_LABELS[final])

    imp = variable_importance(fitted[final], X_test, y_test)

    y_trad = traditional_screening(X_test)
    m_trad = compute_metrics(y_test, y_trad)
    tn, fp, fn, tp = confusion_matrix(y_test, y_trad).ravel()
    m_trad["matriz_confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    ref_df, ref_resumen = reference_case(fitted[final], df)

    versiones = {"scikit-learn": sklearn.__version__}
    if tm.xgboost_available():
        import xgboost
        versiones["xgboost"] = xgboost.__version__

    resultados = {
        "semilla": RANDOM_STATE,
        "versiones": versiones,
        "n_entrenamiento": int(len(y_train)),
        "n_prueba": int(len(y_test)),
        "aptos_prueba": int(y_test.sum()),
        "modelo_final": final,
        "criterio_seleccion": "mayor F1 medio en validación cruzada (Etapa 4)",
        "modelos": {
            name: {
                "entrenamiento": tabla.loc[(name, "entrenamiento")].to_dict(),
                "validacion": {**tabla.loc[(name, "validacion")].to_dict(),
                               "f1_desv": extras[name]["f1_cv_desv"],
                               "auc_roc_desv": extras[name]["auc_cv_desv"]},
                "prueba": tabla.loc[(name, "prueba")].to_dict(),
                "matriz_confusion_prueba": extras[name]["matriz_confusion"],
            } for name in fitted
        },
        "clasificador_trivial_prueba": trivial_baseline(y_train, y_test),
        "cribado_tradicional_prueba": m_trad,
        "importancia_permutacion": {
            v: {"importancia": float(r["importancia"]), "desv": float(r["desv"]),
                "control_negativo": bool(r["control_negativo"])}
            for v, r in imp.iterrows()},
        "caso_referencia": ref_resumen,
    }
    objetos = {"fitted": fitted, "split": split, "tabla": tabla, "extras": extras,
               "importancia": imp, "referencia": ref_df, "y_trad": y_trad}
    return resultados, objetos


def save_results(resultados, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    raiz = os.path.join(_HERE, "..", "..")
    datos = pd.read_csv(os.path.join(raiz, "data", "synthetic", "synthetic_dataset.csv"))
    e4 = tm.load_results(os.path.join(raiz, "results", "etapa4_resultados.json"))
    res, _ = run_evaluation(datos, e4)
    print(json.dumps({k: v for k, v in res.items() if k != "importancia_permutacion"},
                     ensure_ascii=False, indent=2)[:3000])
