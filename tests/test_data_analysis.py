"""
Verificación de la Etapa 2: análisis del conjunto de datos.

Comprueba que el módulo de análisis funciona correctamente y que el dataset
publicado en el repositorio cumple las condiciones necesarias para pasar a la
Etapa 3 de preprocesamiento:

 1. Las funciones de análisis se ejecutan y devuelven resultados con la forma
    esperada.
 2. No existen pares de variables de entrada redundantes.
 3. Ninguna variable aislada determina la etiqueta.
 4. El balance de clases no exige remuestreo.
 5. El dataset supera todas las verificaciones de consistencia física.
 6. Se reproducen las tendencias físicas esperadas.
 7. El rechazo no se concentra en un único criterio.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src", "analysis"))
sys.path.insert(0, os.path.join(HERE, "..", "src", "dataset_generation"))

import data_analysis as da
from generate_dataset import FEATURE_COLUMNS, DERIVED_COLUMNS

RUTA = os.path.join(HERE, "..", "data", "synthetic", "synthetic_dataset.csv")

resultados = []


def check(nombre, condicion):
    resultados.append((nombre, bool(condicion)))


df = pd.read_csv(RUTA)

# ----------------------------------------------------------------------
# 1. Las funciones devuelven resultados con la forma esperada
# ----------------------------------------------------------------------
desc = da.descriptive_statistics(df)
check("descriptive_statistics cubre todas las variables de entrada",
      len(desc) == len(FEATURE_COLUMNS))

por_clase = da.distribution_by_class(df)
check("distribution_by_class calcula la d de Cohen sin valores nulos",
      por_clase["d_de_Cohen"].notna().all())

corr = da.correlation_matrix(df)
check("La matriz de correlacion es cuadrada y simetrica",
      corr.shape[0] == corr.shape[1] == len(FEATURE_COLUMNS)
      and np.allclose(corr.values, corr.values.T, atol=1e-9))

# ----------------------------------------------------------------------
# 2. Sin variables redundantes
# ----------------------------------------------------------------------
red = da.redundant_pairs(df, threshold=0.85)
check("Sin pares de variables de entrada redundantes (|r| < 0.85)", len(red) == 0)

check("La temperatura quedo excluida del conjunto de entrenamiento",
      "T_res_F" not in FEATURE_COLUMNS)
check("La temperatura se conserva en el archivo para trazabilidad",
      "T_res_F" in df.columns)

# ----------------------------------------------------------------------
# 3. Ninguna variable determina la etiqueta por si sola
# ----------------------------------------------------------------------
corr_label = da.correlation_with_label(df)
r_max = corr_label["r_con_etiqueta"].abs().max()
check(f"Ninguna variable aislada determina la etiqueta (|r|max={r_max:.2f})",
      r_max < 0.85)
check("Existe senal aprendible en las variables de entrada", r_max > 0.05)

# ----------------------------------------------------------------------
# 4. Balance de clases
# ----------------------------------------------------------------------
bal = da.class_balance(df)
check(f"Balance de clases sin necesidad de remuestreo "
      f"({bal['proporcion_apto']*100:.1f} % aptos)",
      bal["clase_minoritaria"] >= 0.25)

# ----------------------------------------------------------------------
# 5. Consistencia fisica
# ----------------------------------------------------------------------
cons = da.physical_consistency(df)
n_fallos = int((cons["resultado"] == "FALLA").sum())
check(f"Consistencia fisica del dataset ({len(cons)} verificaciones)",
      n_fallos == 0)

# ----------------------------------------------------------------------
# 6. Tendencias fisicas
# ----------------------------------------------------------------------
mono = da.monotonicity_checks(df)

ea = mono["EA_vs_M"].values
check("El barrido areal decrece al aumentar la movilidad",
      np.all(np.diff(ea) <= 0.02))

ev = mono["EV_vs_Vdp"].values
check("El barrido vertical decrece al aumentar la heterogeneidad",
      np.all(np.diff(ev) <= 0.02))

apto = mono["apto_vs_RF"].values
check("La proporcion de aptos crece con el recobro",
      np.all(np.diff(apto) >= -0.02))

# ----------------------------------------------------------------------
# 7. El rechazo no se concentra en un unico criterio
# ----------------------------------------------------------------------
rech = da.rejection_breakdown(df)
check("El rechazo se reparte entre varios criterios", len(rech) >= 4)
check("Ningun criterio explica por si solo mas del 90 % de los rechazos",
      rech["% de no aptos"].max() < 90.0)

# ----------------------------------------------------------------------
# 8. Separacion entre entradas y derivadas
# ----------------------------------------------------------------------
check("Las derivadas no estan entre las variables de entrenamiento",
      not set(FEATURE_COLUMNS) & set(DERIVED_COLUMNS))

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
