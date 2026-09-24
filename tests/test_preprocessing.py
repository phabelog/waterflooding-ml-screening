"""
Verificación de la Etapa 3: preprocesamiento.

Comprueba que el conjunto queda listo para la Etapa 4 sin fuga de
información:

 1. Limpieza: sin faltantes ni duplicados; los atípicos detectados son
    efecto de la asimetría y desaparecen en escala log10.
 2. División: disjunta, completa, estratificada y reproducible.
 3. Escalado: el transformador se entrega sin ajustar y, una vez ajustado,
    solo aprende del entrenamiento.
 4. Selección: se conservan las variables de entrada definidas en la
    Etapa 1, sin salidas del framework.
 5. Desbalance: la decisión sobre SMOTE usa los mismos umbrales de la
    Etapa 2.
 6. Datos reales: el reporte de dominio detecta faltantes y valores fuera
    de rango.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src", "preprocessing"))
sys.path.insert(0, os.path.join(HERE, "..", "src", "dataset_generation"))

import preprocess as pp
from generate_dataset import FEATURE_COLUMNS, DERIVED_COLUMNS

RUTA = os.path.join(HERE, "..", "data", "synthetic", "synthetic_dataset.csv")
df = pd.read_csv(RUTA)

resultados = []


def check(nombre, condicion):
    resultados.append((nombre, bool(condicion)))


# ----------------------------------------------------------------------
# 1. Limpieza
# ----------------------------------------------------------------------
lim = pp.missing_and_duplicates(df)
check("Sin datos faltantes en las variables de entrada",
      lim["faltantes_total"] == 0)
check("Sin filas duplicadas en las variables de entrada",
      lim["duplicados"] == 0)
check("Los atipicos por IQR se concentran en k y muo (asimetria)",
      set(pp.iqr_outliers(df)) <= set(pp.LOG_FEATURES))
check("Los atipicos desaparecen en escala log10",
      pp.iqr_outliers(df, log_columns=pp.LOG_FEATURES) == {})

# ----------------------------------------------------------------------
# 2. Division estratificada
# ----------------------------------------------------------------------
Xtr, Xte, ytr, yte = pp.split_train_test(df)
check("Proporcion 80 / 20",
      len(Xtr) == int(len(df) * 0.8) and len(Xte) == len(df) - len(Xtr))
check("Entrenamiento y prueba son disjuntos",
      not set(Xtr.index) & set(Xte.index))
check("Entre ambos cubren todo el conjunto",
      set(Xtr.index) | set(Xte.index) == set(df.index))
check("La proporcion de aptos coincide en ambos conjuntos (+/- 1 %)",
      abs(ytr.mean() - yte.mean()) < 0.01)
Xtr2, Xte2, _, _ = pp.split_train_test(df)
check("La division es reproducible (misma semilla, mismos indices)",
      list(Xte.index) == list(Xte2.index))
cv = pp.get_cv()
check("Validacion cruzada estratificada de 5 particiones",
      cv.get_n_splits() == 5)
props = [ytr.iloc[va].mean() for _, va in cv.split(Xtr, ytr)]
check("Cada particion de la validacion cruzada conserva la proporcion",
      max(props) - min(props) < 0.02)

# ----------------------------------------------------------------------
# 3. Escalado sin fuga de informacion
# ----------------------------------------------------------------------
pre = pp.build_preprocessor()
try:
    pre.transform(Xtr)
    sin_ajustar = False
except NotFittedError:
    sin_ajustar = True
check("El preprocesador se entrega sin ajustar", sin_ajustar)

pre_tr = clone(pre).fit(Xtr)
esc = pre_tr.named_transformers_["std"]
medias_tr = Xtr[esc.feature_names_in_].mean().values
medias_todo = df[esc.feature_names_in_].mean().values
check("Las medias del escalado son las del entrenamiento",
      np.allclose(esc.mean_, medias_tr))
check("Y no las del conjunto completo (no hay fuga desde prueba)",
      not np.allclose(esc.mean_, medias_todo))

Z = pre_tr.transform(Xtr)
check("En entrenamiento: media 0 y desviacion 1 por variable",
      np.allclose(Z.mean(axis=0), 0, atol=1e-9)
      and np.allclose(Z.std(axis=0), 1, atol=1e-6))
k_log = pre_tr.named_transformers_["log"].named_steps["escala"]
check("k y muo se estandarizan en escala log10",
      np.allclose(k_log.mean_, np.log10(Xtr[pp.LOG_FEATURES]).mean().values))

# ----------------------------------------------------------------------
# 4. Seleccion de caracteristicas
# ----------------------------------------------------------------------
sel = pp.feature_selection_summary()
check("Se conservan las 16 variables de entrada",
      sel["n_variables"] == 16 and list(Xtr.columns) == FEATURE_COLUMNS)
check("Ninguna salida del framework entra al modelo",
      not set(Xtr.columns) & set(DERIVED_COLUMNS))
check("La temperatura sigue excluida (Etapa 2)",
      "T_res_F" not in Xtr.columns)
check("Los controles negativos forman parte de las entradas",
      set(pp.NEGATIVE_CONTROLS) <= set(Xtr.columns))

# ----------------------------------------------------------------------
# 5. Desbalance
# ----------------------------------------------------------------------
des = pp.resampling_decision(ytr)
check("Con desbalance leve no se aplica SMOTE",
      des["usar_smote"] is False)
check("scale_pos_weight = no aptos / aptos del entrenamiento",
      abs(des["scale_pos_weight"] - (ytr == 0).sum() / (ytr == 1).sum()) < 1e-12)
check("Con clase minoritaria < 25 % si se activaria SMOTE",
      pp.resampling_decision(np.r_[np.ones(10), np.zeros(90)])["usar_smote"])

# ----------------------------------------------------------------------
# 6. Reporte de dominio para los datos reales
# ----------------------------------------------------------------------
nuevo = Xte.head(5).copy()
nuevo.iloc[0, nuevo.columns.get_loc("k_md")] = np.nan
nuevo.iloc[1, nuevo.columns.get_loc("Vdp")] = 0.99
rep, completos = pp.domain_report(nuevo, Xtr)
check("El reporte de dominio detecta un dato faltante",
      rep.loc["k_md", "faltantes"] == 1)
check("El reporte de dominio detecta un valor fuera de rango",
      rep.loc["Vdp", "fuera_rango"] == 1)
check("Cuenta solo los registros completos", completos == 4)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"{'Verificacion':<66}{'Resultado'}")
    print("-" * 78)
    for nombre, ok in resultados:
        print(f"{nombre:<66}{'PASA' if ok else 'FALLA'}")
    n_ok = sum(ok for _, ok in resultados)
    print("-" * 78)
    print(f"{n_ok}/{len(resultados)} verificaciones superadas.")
    if n_ok != len(resultados):
        sys.exit(1)
