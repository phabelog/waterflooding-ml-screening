"""
Verificación de la Etapa 5: evaluación y validación.

 1. La prueba no interviene en el ajuste ni en la selección del modelo.
 2. Las métricas son coherentes (rango, matriz de confusión, AUC).
 3. El modelo final es el de mayor F1 en validación cruzada.
 4. Los controles negativos no tienen importancia.
 5. El cribado tradicional por rangos no descarta ningún escenario apto.
 6. El caso de referencia queda dentro del dominio de entrenamiento.
 7. Si ya existe results/etapa5_resultados.json, su contenido es coherente.

Para que la prueba sea rápida se usa el árbol de decisión (y XGBoost solo
en las comprobaciones del archivo de resultados).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src", "evaluation"))

import evaluate as ev
import preprocess as pp
import train_models as tm

RAIZ = os.path.join(HERE, "..")
df = pd.read_csv(os.path.join(RAIZ, "data", "synthetic", "synthetic_dataset.csv"))
e4 = tm.load_results(os.path.join(RAIZ, "results", "etapa4_resultados.json"))
RESULTADOS = os.path.join(RAIZ, "results", "etapa5_resultados.json")

resultados, pendientes = [], []


def check(nombre, condicion):
    resultados.append((nombre, bool(condicion)))


# ----------------------------------------------------------------------
# 1. Ajuste final sin fuga desde la prueba
# ----------------------------------------------------------------------
fitted, split = ev.fit_final_models(df, e4, models=["arbol_decision"])
X_train, X_test, y_train, y_test = split
modelo = fitted["arbol_decision"]

check("Division 4 000 / 1 000 de la Etapa 3", len(X_train) == 4000 and len(X_test) == 1000)
check("Entrenamiento y prueba son disjuntos", len(set(X_train.index) & set(X_test.index)) == 0)
check("El modelo usa los hiperparametros guardados en la Etapa 4",
      all(modelo.named_steps["clf"].get_params()[k] == v
          for k, v in e4["modelos"]["arbol_decision"]["mejores_hiperparametros"].items()))
check("El escalado se ajusto solo con el entrenamiento",
      modelo.named_steps["pre"].n_features_in_ == 16)

# ----------------------------------------------------------------------
# 2. Métricas
# ----------------------------------------------------------------------
tabla, extras = ev.evaluate_models(fitted, split, e4)
fila = tabla.loc[("arbol_decision", "prueba")]
c = extras["arbol_decision"]["matriz_confusion"]
check("Todas las metricas estan en [0, 1]", bool(((tabla >= 0) & (tabla <= 1)).all().all()))
check("La matriz de confusion suma los 1 000 casos de prueba", sum(c.values()) == 1000)
check("Precision y recall coinciden con la matriz de confusion",
      abs(fila["precision"] - c["tp"] / (c["tp"] + c["fp"])) < 1e-9
      and abs(fila["recall"] - c["tp"] / (c["tp"] + c["fn"])) < 1e-9)
check("La validacion repite el F1 de la Etapa 4",
      abs(tabla.loc[("arbol_decision", "validacion"), "f1"]
          - e4["modelos"]["arbol_decision"]["validacion_cruzada"]["f1_media"]) < 1e-12)
check("AUC-ROC en prueba muy superior al azar (> 0.9)", fila["auc_roc"] > 0.9)
check("El F1 de prueba no cae mas de 0.05 respecto de la validacion",
      tabla.loc[("arbol_decision", "validacion"), "f1"] - fila["f1"] < 0.05)
triv = ev.trivial_baseline(y_train, y_test)
check("El modelo supera al clasificador trivial en la prueba", fila["f1"] > triv["f1"] + 0.1)

# ----------------------------------------------------------------------
# 3. Selección del modelo final
# ----------------------------------------------------------------------
check("El modelo final es el de mayor F1 en validacion cruzada (XGBoost)",
      ev.select_final_model(e4) == "xgboost")

# ----------------------------------------------------------------------
# 4. Importancia y controles negativos
# ----------------------------------------------------------------------
imp = ev.variable_importance(modelo, X_test, y_test, n_repeats=10)
neg = imp[imp["control_negativo"]]
check("Los controles negativos son porosidad y profundidad",
      set(neg.index) == {"phi", "depth_ft"})
check("La importancia de los controles negativos es despreciable (< 0.01)",
      bool((neg["importancia"].abs() < 0.01).all()))
check("La variable mas importante pertenece al modelo fisico",
      not imp.iloc[0]["control_negativo"])

# ----------------------------------------------------------------------
# 5. Cribado tradicional por rangos
# ----------------------------------------------------------------------
y_trad = ev.traditional_screening(X_test)
m_trad = ev.compute_metrics(y_test, y_trad)
check("El cribado por rangos no descarta ningun escenario apto (recall = 1)",
      m_trad["recall"] == 1.0)
check("El cribado por rangos acepta escenarios no aptos (precision < 1)",
      m_trad["precision"] < 1.0)
check("El modelo supera en F1 al cribado por rangos", fila["f1"] > m_trad["f1"])

# ----------------------------------------------------------------------
# 6. Caso de referencia
# ----------------------------------------------------------------------
ref, res_ref = ev.reference_case(modelo, df, n=300)
check("Las propiedades medidas del caso de referencia quedan fijas",
      all(ref[k].nunique() == 1 for k in ev.REFERENCE_FIXED))
check("El caso de referencia queda dentro del dominio de entrenamiento",
      res_ref["realizaciones_fuera_de_dominio"] == 0)
check("El caso de referencia es reproducible (misma semilla)",
      ev.reference_case_scenarios(50).equals(ev.reference_case_scenarios(50)))
check("Modelo y criterio fisico coinciden en mas del 85 % de las realizaciones",
      res_ref["coincidencia_modelo_fisica"] > 0.85)

# ----------------------------------------------------------------------
# 7. Archivo de resultados
# ----------------------------------------------------------------------
if os.path.exists(RESULTADOS):
    with open(RESULTADOS, encoding="utf-8") as f:
        d = json.load(f)
    check("El archivo de resultados contiene los cuatro modelos",
          set(d["modelos"]) == set(tm.MODEL_NAMES))
    check("El modelo final del archivo es XGBoost", d["modelo_final"] == "xgboost")
    check("El arbol del archivo se reproduce con el codigo actual",
          abs(d["modelos"]["arbol_decision"]["prueba"]["f1"] - fila["f1"]) < 0.01)
    check("Cada matriz de confusion del archivo suma 1 000",
          all(sum(m["matriz_confusion_prueba"].values()) == 1000 for m in d["modelos"].values()))
else:
    pendientes.append("Resultados: aun no se sube results/etapa5_resultados.json")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"{'Verificacion':<72}{'Resultado'}")
    print("-" * 84)
    for nombre, ok in resultados:
        print(f"{nombre:<72}{'PASA' if ok else 'FALLA'}")
    for p_ in pendientes:
        print(f"{p_:<72}PENDIENTE")
    n_ok = sum(ok for _, ok in resultados)
    print("-" * 84)
    print(f"{n_ok}/{len(resultados)} verificaciones superadas.")
    if n_ok != len(resultados):
        sys.exit(1)
