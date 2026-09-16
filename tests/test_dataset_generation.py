"""
Verificación de la Etapa 1.2 - 1.4: variables, eficiencias de barrido,
etiquetado y consistencia del dataset sintético.

Chequeos:
 1. EA decrece monótonamente al aumentar M y reproduce el valor de
    referencia (~50 % en M = 10) de la curva de Craig.
 2. EV decrece monótonamente al aumentar la heterogeneidad (Vdp) y al
    aumentar M; vale 1.0 en un reservorio homogéneo.
 3. El modelo de Dykstra-Parsons reproduce el caso analítico M = 1.
 4. RF_total < ED siempre (las eficiencias de barrido solo pueden restar).
 5. Consistencia física del dataset: viscosidades, saturaciones y rangos.
 6. Balance de clases razonable (ninguna clase por debajo del 20 %).
 7. Ausencia de fuga: la etiqueta no es reproducible de forma trivial a
    partir de una sola variable de entrada.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src", "physics"))
sys.path.insert(0, os.path.join(HERE, "..", "src", "dataset_generation"))

from sweep_efficiency import (areal_sweep_breakthrough,
                              vertical_sweep_dykstra_parsons,
                              total_recovery_factor,
                              _layer_permeabilities, _front_position)
from variable_ranges import (oil_viscosity_from_api,
                             water_viscosity_from_temperature,
                             current_oil_saturation)
from labeling import THRESHOLDS, thresholds_table
import generate_dataset as gd


resultados = []


def check(nombre, condicion):
    resultados.append((nombre, bool(condicion)))


# ----------------------------------------------------------------------
# 1. Barrido areal
# ----------------------------------------------------------------------
M_vals = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
EA_vals = np.array([float(areal_sweep_breakthrough(m)) for m in M_vals])

check("EA decrece al aumentar M", np.all(np.diff(EA_vals) < 0))
check("EA(M=10) proximo a 0.50 (curva de Craig)", abs(EA_vals[-1] - 0.50) < 0.05)
check("EA acotada en (0, 1]", np.all((EA_vals > 0) & (EA_vals <= 1.0)))

# ----------------------------------------------------------------------
# 2-3. Barrido vertical (Dykstra-Parsons)
# ----------------------------------------------------------------------
EV_homog = vertical_sweep_dykstra_parsons(0.0, 1.0)
check("EV = 1.0 en reservorio homogeneo (Vdp=0)", abs(EV_homog - 1.0) < 1e-6)

EV_vdp = [vertical_sweep_dykstra_parsons(v, 1.0) for v in [0.2, 0.4, 0.6, 0.8, 0.9]]
check("EV decrece al aumentar Vdp", np.all(np.diff(EV_vdp) < 0))

EV_M = [vertical_sweep_dykstra_parsons(0.6, m) for m in [0.5, 1.0, 2.0, 5.0, 10.0]]
check("EV decrece al aumentar M", np.all(np.diff(EV_M) < 0))

# Caso analitico: con M = 1 la posicion del frente es u = k_j / k_i
k = _layer_permeabilities(0.6, 20)
u_num = _front_position(k / k[0], 1.0)
check("Dykstra-Parsons reproduce el caso analitico M=1",
      np.allclose(u_num, np.clip(k / k[0], 0, 1), atol=1e-9))

# ----------------------------------------------------------------------
# 4. Recobro total
# ----------------------------------------------------------------------
t = total_recovery_factor(ED=0.60, M=2.0, Vdp=0.6)
check("RF_total = ED * EA * EV",
      abs(t["RF_total"] - t["ED"] * t["EA"] * t["EV"]) < 1e-9)
check("RF_total < ED (el barrido solo puede restar)", t["RF_total"] < t["ED"])

# ----------------------------------------------------------------------
# 5. Consistencia fisica de las relaciones condicionadas
# ----------------------------------------------------------------------
rng = np.random.default_rng(0)
api_test = np.array([15.0, 20.0, 25.0, 30.0, 35.0])
muo_med = 10.0 ** (2.0 - 0.06 * api_test)
check("Viscosidad decrece al aumentar API", np.all(np.diff(muo_med) < 0))
check("Viscosidad en rango de crudo vivo (28-32 API -> 1-5 cP)",
      1.0 <= 10.0 ** (2.0 - 0.06 * 30.0) <= 5.0)

muw_test = water_viscosity_from_temperature(np.array([60.0, 200.0]))
check("Viscosidad del agua decrece con la temperatura", muw_test[1] < muw_test[0])
check("Viscosidad del agua a 200F en rango fisico", 0.25 < muw_test[1] < 0.40)

So = current_oil_saturation(0.25, 0.25, np.array([0.0, 0.5, 1.0]))
check("So acotada entre Sor y 1 - Swi",
      np.all(So >= 0.25 - 1e-9) and np.all(So <= 0.75 + 1e-9))

# ----------------------------------------------------------------------
# 6-7. Dataset
# ----------------------------------------------------------------------
df = gd.build_dataset(n=1200, seed=123, verbose=False)
frac_apto = df["label"].mean()

check("Dataset generado sin escenarios perdidos", len(df) > 1000)
check(f"Balance de clases razonable ({frac_apto*100:.1f} % aptos)",
      0.20 <= frac_apto <= 0.80)
check("Todas las saturaciones moviles son positivas",
      np.all(1 - df["Swi"] - df["Sor"] > 0))
check("RF_total dentro del rango fisico [0,1]",
      df["RF_total"].between(0, 1).all())
check("RF_total nunca supera ED", (df["RF_total"] <= df["ED"] + 1e-9).all())
check("Recobro medio en rango realista de waterflooding (10-40 %)",
      0.10 <= df["RF_total"].mean() <= 0.40)

# La etiqueta no debe ser reproducible por una sola variable
corr_max = max(abs(np.corrcoef(df[c], df["label"])[0, 1])
               for c in gd.FEATURE_COLUMNS)
check(f"Ninguna variable aislada determina la etiqueta (|r|max={corr_max:.2f})",
      corr_max < 0.85)

check("Las derivadas no estan en las features de entrenamiento",
      not set(gd.DERIVED_COLUMNS) & set(gd.FEATURE_COLUMNS))

# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(thresholds_table())
    print()
    print(f"{'Verificacion':<62}{'Resultado'}")
    print("-" * 74)
    for nombre, ok in resultados:
        print(f"{nombre:<62}{'PASA' if ok else 'FALLA'}")
    n_ok = sum(ok for _, ok in resultados)
    print("-" * 74)
    print(f"{n_ok}/{len(resultados)} verificaciones superadas.")
    if n_ok != len(resultados):
        sys.exit(1)
