"""
Verificación de la Etapa 4: desarrollo de modelos.

 1. Los cuatro modelos están definidos y comparten el transformador de la
    Etapa 3 dentro de un pipeline.
 2. La ponderación de clases de la Etapa 3 se aplica a los cuatro.
 3. Todas las combinaciones de las grillas son válidas.
 4. La búsqueda usa las mismas 5 particiones de la Etapa 3 y solo el
    conjunto de entrenamiento (sin fuga desde la prueba).
 5. La búsqueda es reproducible y el modelo base supera a un clasificador
    trivial.
 6. Si ya existe results/etapa4_resultados.json, su contenido es coherente.

Las verificaciones de XGBoost se omiten si la librería no está instalada
(en Google Colab viene instalada).
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src", "models"))

import train_models as tm
import preprocess as pp

df = pd.read_csv(os.path.join(HERE, "..", "data", "synthetic", "synthetic_dataset.csv"))
RESULTADOS = os.path.join(HERE, "..", "results", "etapa4_resultados.json")
HAY_XGB = tm.xgboost_available()

X_train, X_test, y_train, y_test = pp.split_train_test(df)
SPW = pp.resampling_decision(y_train)["scale_pos_weight"]
p = y_train.mean()
F1_TRIVIAL = 2 * p / (1 + p)          # F1 de declarar "apto" a todos los casos

resultados, omitidas, pendientes = [], [], []


def check(nombre, condicion):
    resultados.append((nombre, bool(condicion)))


modelos = [m for m in tm.MODEL_NAMES if m != "xgboost" or HAY_XGB]
if not HAY_XGB:
    omitidas.append("Verificaciones de XGBoost (xgboost no instalado)")

# ----------------------------------------------------------------------
# 1. Definición de los modelos
# ----------------------------------------------------------------------
check("Los cuatro modelos de la metodologia estan definidos",
      tm.MODEL_NAMES == ["arbol_decision", "random_forest", "xgboost", "svm"])
check("Cada modelo tiene su grilla de hiperparametros",
      set(tm.PARAM_GRIDS) == set(tm.MODEL_NAMES))

pipes = {m: tm.build_model(m, SPW) for m in modelos}
ok_pre = True
for m, pipe in pipes.items():
    pre = pipe.named_steps["pre"]
    try:
        pre.transform(X_train)
        ok_pre = False
    except NotFittedError:
        pass
    ok_pre &= isinstance(pre, ColumnTransformer) and list(pipe.named_steps) == ["pre", "clf"]
check("Todos usan el transformador de la Etapa 3, sin ajustar", ok_pre)

# ----------------------------------------------------------------------
# 2. Ponderación de clases
# ----------------------------------------------------------------------
check("Arbol, Random Forest y SVM usan class_weight='balanced'",
      all(pipes[m].named_steps["clf"].class_weight == "balanced"
          for m in ("arbol_decision", "random_forest", "svm")))
if HAY_XGB:
    check("XGBoost usa scale_pos_weight de la Etapa 3",
          abs(pipes["xgboost"].named_steps["clf"].get_params()["scale_pos_weight"] - SPW) < 1e-12)

# ----------------------------------------------------------------------
# 3. Grillas válidas
# ----------------------------------------------------------------------
ok_grid = True
for m in modelos:
    g = tm.PARAM_GRIDS[m]
    for combo in itertools.product(*g.values()):
        try:
            tm.build_model(m, SPW).set_params(**dict(zip(g.keys(), combo)))
        except ValueError:
            ok_grid = False
check("Todas las combinaciones de las grillas son parametros validos", ok_grid)
check("La optimizacion se guia por F1 y registra precision, recall y accuracy",
      tm.REFIT_METRIC == "f1"
      and set(tm.SCORING) == {"f1", "precision", "recall", "accuracy"})

# ----------------------------------------------------------------------
# 4. Búsqueda sin fuga, sobre las particiones de la Etapa 3
# ----------------------------------------------------------------------
s1 = tm.tune_model("arbol_decision", X_train, y_train, SPW, n_jobs=1)
check("La busqueda usa 5 particiones", s1.n_splits_ == 5)
mismas = all(np.array_equal(a[1], b[1]) for a, b in
             zip(s1.cv.split(X_train, y_train), pp.get_cv().split(X_train, y_train)))
check("Son las mismas particiones definidas en la Etapa 3", mismas)
esc = s1.best_estimator_.named_steps["pre"].named_transformers_["std"]
check("El modelo final de la busqueda vio solo el entrenamiento (4 000 casos)",
      int(esc.n_samples_seen_) == len(X_train) == 4000)
check("Ningun indice de prueba aparece en el entrenamiento",
      not set(X_test.index) & set(X_train.index))

# ----------------------------------------------------------------------
# 5. Reproducibilidad y desempeño mínimo
# ----------------------------------------------------------------------
s2 = tm.tune_model("arbol_decision", X_train, y_train, SPW, n_jobs=1)
check("La busqueda es reproducible (mismos hiperparametros y F1)",
      s1.best_params_ == s2.best_params_ and s1.best_score_ == s2.best_score_)
check(f"El modelo base supera al clasificador trivial (F1 > {F1_TRIVIAL:.3f})",
      s1.best_score_ > F1_TRIVIAL + 0.1)

res_arbol = tm.summarize_search("arbol_decision", s1)
mejores = {"arbol_decision": tm._clean_params(s1.best_params_)}
tmp = os.path.join(HERE, "_tmp_etapa4.json")
tm.save_results(pd.DataFrame([res_arbol]).set_index("modelo"), mejores, tmp)
reconstruido = tm.build_best_model(
    "arbol_decision", tm.load_results(tmp)["modelos"]["arbol_decision"]["mejores_hiperparametros"], SPW)
os.remove(tmp)
check("Los hiperparametros guardados reconstruyen el mismo modelo",
      all(reconstruido.get_params()[f"clf__{k}"] == v for k, v in mejores["arbol_decision"].items()))

# ----------------------------------------------------------------------
# 6. Resultados de la Etapa 4 (si ya fueron generados en Colab)
# ----------------------------------------------------------------------
if os.path.exists(RESULTADOS):
    d = tm.load_results(RESULTADOS)
    check("El archivo de resultados contiene los cuatro modelos",
          set(d["modelos"]) == set(tm.MODEL_NAMES))
    en_grilla = True
    for m, info in d["modelos"].items():
        for k, v in info["mejores_hiperparametros"].items():
            en_grilla &= v in tm.PARAM_GRIDS[m][f"clf__{k}"]
    check("Los mejores hiperparametros pertenecen a sus grillas", en_grilla)
    metricas_ok = all(0 <= info["validacion_cruzada"][f"{k}_media"] <= 1
                      for info in d["modelos"].values() for k in tm.SCORING)
    check("Las metricas de validacion cruzada estan en [0, 1]", metricas_ok)
    check("Los cuatro modelos superan al clasificador trivial",
          all(info["validacion_cruzada"]["f1_media"] > F1_TRIVIAL + 0.1
              for info in d["modelos"].values()))
    arbol = d["modelos"]["arbol_decision"]
    check("El arbol guardado se reproduce con el codigo actual",
          arbol["mejores_hiperparametros"] == mejores["arbol_decision"]
          and abs(arbol["validacion_cruzada"]["f1_media"] - s1.best_score_) < 0.01)
else:
    pendientes.append("Resultados: aun no se sube results/etapa4_resultados.json")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"{'Verificacion':<72}{'Resultado'}")
    print("-" * 84)
    for nombre, ok in resultados:
        print(f"{nombre:<72}{'PASA' if ok else 'FALLA'}")
    for o in omitidas:
        print(f"{o:<72}OMITIDA")
    for p_ in pendientes:
        print(f"{p_:<72}PENDIENTE")
    n_ok = sum(ok for _, ok in resultados)
    print("-" * 84)
    print(f"{n_ok}/{len(resultados)} verificaciones superadas.")
    if n_ok != len(resultados):
        sys.exit(1)
