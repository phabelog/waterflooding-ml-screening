"""
Desarrollo de modelos de clasificación.

Corresponde a la Etapa 4 del flujo de trabajo metodológico:

    4.1  Árbol de decisión como modelo base
    4.2  Random Forest
    4.3  XGBoost
    4.4  SVM
    4.5  Optimización de hiperparámetros mediante validación cruzada
         estratificada

Todo lo que este módulo utiliza viene fijado por la Etapa 3:
  - la división entrenamiento / prueba  (pp.split_train_test)
  - las 5 particiones de validación      (pp.get_cv)
  - el transformador de escalado         (pp.build_preprocessor)
  - la ponderación de clases             (pp.resampling_decision)

El conjunto de PRUEBA no se usa en ningún punto de esta etapa. La
optimización se realiza solo con el entrenamiento, y la comparación y
selección del modelo final corresponden a la Etapa 5.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

_HERE = os.path.dirname(__file__)
for _sub in ("preprocessing", "analysis", "dataset_generation"):
    sys.path.insert(0, os.path.join(_HERE, "..", _sub))

import preprocess as pp  # noqa: E402

RANDOM_STATE = pp.RANDOM_STATE

MODEL_NAMES = ["arbol_decision", "random_forest", "xgboost", "svm"]

MODEL_LABELS = {
    "arbol_decision": "Árbol de decisión (base)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svm": "SVM",
}

# Métrica que guía la optimización. El objetivo específico 4 del
# anteproyecto fija precisión, recall y F1; se optimiza F1 porque equilibra
# los dos errores del cribado: descartar un buen candidato (falso negativo)
# y recomendar uno inviable (falso positivo). Las demás se registran.
REFIT_METRIC = "f1"
SCORING = {
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
}

# ----------------------------------------------------------------------
#  Grillas de hiperparámetros
# ----------------------------------------------------------------------
# Se mantienen deliberadamente acotadas: cubren los hiperparámetros que
# controlan la complejidad de cada modelo, con valores habituales en la
# literatura, sin multiplicar el costo de cómputo.
PARAM_GRIDS = {
    # Profundidad y tamaño mínimo de hoja controlan el sobreajuste del árbol.
    "arbol_decision": {
        "clf__max_depth": [5, 8, 12, 16, None],
        "clf__min_samples_leaf": [5, 20, 50],
    },
    # El número de árboles se fija en 300: más árboles no sobreajustan, solo
    # aumentan el costo. Los árboles crecen sin límite de profundidad, como es
    # habitual en Random Forest; se ajustan el tamaño mínimo de hoja y la
    # fracción de variables evaluadas en cada división.
    "random_forest": {
        "clf__min_samples_leaf": [1, 3],
        "clf__max_features": ["sqrt", 0.5, 0.8],
    },
    # Número de árboles, profundidad y tasa de aprendizaje gobiernan el
    # compromiso sesgo-varianza en boosting.
    "xgboost": {
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.05, 0.1],
    },
    # C controla la penalización de errores y gamma el alcance del núcleo
    # RBF (con 16 variables estandarizadas, el valor por defecto es 1/16).
    "svm": {
        "clf__C": [10, 100, 1000],
        "clf__gamma": [0.005, 0.01, 0.02],
    },
}


def xgboost_available():
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def build_model(name, scale_pos_weight=1.0):
    """
    Pipeline sin ajustar: transformador de la Etapa 3 + clasificador.

    El escalado forma parte del pipeline para que, dentro de la validación
    cruzada, se ajuste solo con la parte de ajuste de cada partición.
    """
    if name == "arbol_decision":
        clf = DecisionTreeClassifier(class_weight="balanced",
                                     random_state=RANDOM_STATE)
    elif name == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=RANDOM_STATE, n_jobs=1)
    elif name == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=scale_pos_weight,
                            eval_metric="logloss", tree_method="hist",
                            random_state=RANDOM_STATE, n_jobs=1)
    elif name == "svm":
        # probability=False durante la búsqueda: estimar probabilidades en
        # SVC exige una validación interna adicional que multiplica el costo.
        # Si la SVM resultara seleccionada en la Etapa 5, se reajusta con
        # probability=True para la interfaz de la Etapa 6.
        clf = SVC(kernel="rbf", class_weight="balanced",
                  random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Modelo desconocido: {name}")

    return Pipeline([("pre", pp.build_preprocessor()), ("clf", clf)])


def tune_model(name, X_train, y_train, scale_pos_weight=1.0, n_jobs=-1):
    """Búsqueda en grilla con las 5 particiones estratificadas de la Etapa 3."""
    search = GridSearchCV(
        build_model(name, scale_pos_weight),
        PARAM_GRIDS[name],
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=pp.get_cv(),
        return_train_score=True,
        n_jobs=n_jobs,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    search.elapsed_s_ = time.time() - t0
    return search


def summarize_search(name, search):
    """Métricas de validación cruzada en la mejor combinación."""
    cv = search.cv_results_
    i = search.best_index_
    fila = {"modelo": name, "etiqueta": MODEL_LABELS[name]}
    for m in SCORING:
        fila[f"{m}_media"] = float(cv[f"mean_test_{m}"][i])
        fila[f"{m}_desv"] = float(cv[f"std_test_{m}"][i])
    fila["f1_entrenamiento"] = float(cv["mean_train_f1"][i])
    fila["brecha_f1"] = fila["f1_entrenamiento"] - fila["f1_media"]
    fila["combinaciones"] = int(len(cv["params"]))
    fila["tiempo_s"] = round(float(getattr(search, "elapsed_s_", np.nan)), 1)
    return fila


def _clean_params(params):
    """Quita el prefijo 'clf__' y convierte a tipos serializables."""
    out = {}
    for k, v in params.items():
        k = k.replace("clf__", "")
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        out[k] = v
    return out


def run_all(df, models=None, n_jobs=-1, verbose=True):
    """
    Optimiza los modelos indicados (por defecto, los cuatro) sobre el
    conjunto de ENTRENAMIENTO de la Etapa 3.

    Returns
    -------
    resumen : DataFrame con las métricas de validación cruzada
    mejores : dict modelo -> mejores hiperparámetros
    busquedas : dict modelo -> objeto GridSearchCV ajustado
    """
    models = models or MODEL_NAMES
    X_train, _, y_train, _ = pp.split_train_test(df)   # la prueba se descarta
    spw = pp.resampling_decision(y_train)["scale_pos_weight"]

    filas, mejores, busquedas = [], {}, {}
    for name in models:
        if name == "xgboost" and not xgboost_available():
            raise ImportError("xgboost no está instalado: pip install xgboost")
        if verbose:
            n = int(np.prod([len(v) for v in PARAM_GRIDS[name].values()]))
            print(f"{MODEL_LABELS[name]:26s} {n:2d} combinaciones x 5 particiones ...",
                  end=" ", flush=True)
        s = tune_model(name, X_train, y_train, spw, n_jobs)
        busquedas[name] = s
        mejores[name] = _clean_params(s.best_params_)
        filas.append(summarize_search(name, s))
        if verbose:
            print(f"F1 = {s.best_score_:.4f}  ({s.elapsed_s_:.0f} s)")

    resumen = pd.DataFrame(filas).set_index("modelo")
    return resumen, mejores, busquedas


def save_results(resumen, mejores, path):
    """
    Guarda los mejores hiperparámetros y las métricas de validación
    cruzada. La Etapa 5 reconstruye cada modelo a partir de este archivo,
    lo ajusta con todo el entrenamiento y lo evalúa una única vez sobre la
    prueba.
    """
    versiones = {"scikit-learn": sklearn.__version__}
    if xgboost_available():
        import xgboost
        versiones["xgboost"] = xgboost.__version__
    data = {
        "metrica_de_optimizacion": REFIT_METRIC,
        "particiones_cv": pp.CV_FOLDS,
        "semilla": RANDOM_STATE,
        "versiones": versiones,
        "modelos": {
            name: {"mejores_hiperparametros": mejores[name],
                   "validacion_cruzada": {k: (v.item() if hasattr(v, "item") else v)
                                          for k, v in resumen.loc[name].items()
                                          if k != "etiqueta"}}
            for name in resumen.index
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_results(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_best_model(name, best_params, scale_pos_weight=1.0):
    """Reconstruye un pipeline con los hiperparámetros guardados (Etapa 5)."""
    pipe = build_model(name, scale_pos_weight)
    pipe.set_params(**{f"clf__{k}": v for k, v in best_params.items()})
    return pipe
