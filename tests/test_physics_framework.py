"""
Verificación del framework físico con un caso de control.

Chequeos:
1. Condiciones de frontera de fw(Sw): fw(Swi)=0, fw(1-Sor)=1, monotonicidad.
2. Consistencia cruzada de los dos métodos numéricos de localización del
   frente de choque (Swf).
3. Consistencia entre balance de materia y construcción de Welge para
   Sw_avg a la irrupción.
4. Rangos físicos válidos: 0 <= Swf, Sw_avg_BT <= 1-Sor ; 0 <= RF_BT <= 1.
5. Comportamiento esperado: al aumentar M (relación de movilidad,
   empeorando el desplazamiento), el RF a la irrupción debe DISMINUIR.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "physics"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from corey import relative_permeabilities
from buckley_leverett import welge_results
from darcy import endpoint_mobility_ratio, radial_injectivity_index

# ------------------------------------------------------------------
# Caso base de control (valores típicos de arenisca consolidada,
# reservorio candidato "razonable" a waterflooding)
# ------------------------------------------------------------------
base_case = dict(
    Swi=0.20, Sor=0.25,
    kro_max=0.80, krw_max=0.30,
    no=2.0, nw=2.0,
    muo=5.0, muw=1.0,   # cP
)

params_fw = dict(
    Swi=base_case["Swi"], Sor=base_case["Sor"],
    kro_max=base_case["kro_max"], krw_max=base_case["krw_max"],
    no=base_case["no"], nw=base_case["nw"],
    muo=base_case["muo"], muw=base_case["muw"],
)

print("=" * 70)
print("CASO BASE")
print("=" * 70)
for k, v in base_case.items():
    print(f"  {k:10s} = {v}")

res = welge_results(params_fw, base_case["Swi"], base_case["Sor"])

print("\n--- Resultados Buckley-Leverett / Welge ---")
print(f"  Swf (frente de choque)      = {res['Swf']:.4f}")
print(f"  fwf (corte de agua en Swf)  = {res['fwf']:.4f}")
print(f"  Pendiente tangente          = {res['slope']:.4f}")
print(f"  Diferencia entre métodos A/B (Swf) = {res['cross_check_diff']:.2e}  (debe ser ~0)")
print(f"  Qi_BT (PV inyectados a irrupción)  = {res['Qi_BT']:.4f}")
print(f"  Sw_avg a la irrupción       = {res['Sw_avg_BT']:.4f}")
print(f"  RF a la irrupción           = {res['RF_BT']*100:.2f} %")

# ------------------------------------------------------------------
# Checks automáticos
# ------------------------------------------------------------------
errors = []

Sw_test = np.array([base_case["Swi"], 1 - base_case["Sor"]])
kro_t, krw_t = relative_permeabilities(Sw_test, base_case["Swi"], base_case["Sor"],
                                        base_case["kro_max"], base_case["krw_max"],
                                        base_case["no"], base_case["nw"])
if not np.isclose(krw_t[0], 0.0, atol=1e-8):
    errors.append("krw(Swi) != 0")
if not np.isclose(kro_t[1], 0.0, atol=1e-8):
    errors.append("kro(1-Sor) != 0")

fw_curve = res["fw_curve"]
if not (fw_curve["fw"][0] < 0.05):
    errors.append("fw(Swi) no es ~0")
if not (fw_curve["fw"][-1] > 0.95):
    errors.append("fw(1-Sor) no es ~1")
if not np.all(np.diff(fw_curve["fw"]) >= -1e-6):
    errors.append("fw(Sw) no es monótona creciente")

if res["cross_check_diff"] > 5e-3:
    errors.append("Los dos métodos de localización del frente de choque no coinciden")

if not (base_case["Swi"] < res["Swf"] < 1 - base_case["Sor"]):
    errors.append("Swf fuera del rango físico (Swi, 1-Sor)")

if not (res["Swf"] <= res["Sw_avg_BT"] <= 1 - base_case["Sor"]):
    errors.append("Sw_avg_BT fuera de rango [Swf, 1-Sor]")

if not (0.0 <= res["RF_BT"] <= 1.0):
    errors.append("RF_BT fuera de rango [0,1]")

# --- Sensibilidad física: M mayor (más desfavorable) => RF_BT menor ---
def rf_bt_for_muo(muo_val):
    p = dict(params_fw)
    p["muo"] = muo_val
    r = welge_results(p, base_case["Swi"], base_case["Sor"])
    return r["RF_BT"]

rf_low_visc = rf_bt_for_muo(2.0)   # petróleo menos viscoso -> M más favorable
rf_high_visc = rf_bt_for_muo(20.0)  # petróleo más viscoso -> M más desfavorable
print(f"\n--- Sensibilidad física (debe cumplirse: RF con muo=2cP > RF con muo=20cP) ---")
print(f"  RF_BT (muo=2 cP)  = {rf_low_visc*100:.2f} %")
print(f"  RF_BT (muo=20 cP) = {rf_high_visc*100:.2f} %")
if not (rf_low_visc > rf_high_visc):
    errors.append("Sensibilidad física incorrecta: mayor viscosidad de petróleo no redujo el RF a la irrupción")

# --- Darcy: relación de movilidad e inyectividad ---
M = endpoint_mobility_ratio(base_case["krw_max"], base_case["muw"],
                             base_case["kro_max"], base_case["muo"])
print(f"\n--- Darcy ---")
print(f"  Relación de movilidad end-point M = {M:.3f}  ({'favorable' if M < 1 else 'desfavorable'})")

II = radial_injectivity_index(k_md=200, h_ft=40, muw_cp=1.0, re_ft=1000, rw_ft=0.33, skin=0)
print(f"  Índice de inyectividad (k=200mD, h=40ft, re=1000ft) = {II:.3f} bbl/d/psi")
if II <= 0:
    errors.append("Índice de inyectividad no positivo")

# ------------------------------------------------------------------
print("\n" + "=" * 70)
if errors:
    print("RESULTADO: FALLARON", len(errors), "VERIFICACIONES:")
    for e in errors:
        print("  - " + e)
else:
    print("RESULTADO: TODAS LAS VERIFICACIONES PASARON CORRECTAMENTE ✔")
print("=" * 70)

# ------------------------------------------------------------------
# Gráfico: curva de flujo fraccional + construcción de Welge
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(fw_curve["Sw"], fw_curve["fw"], color="#1f4e79", lw=2, label="fw(Sw)")
Sw_tan = np.array([base_case["Swi"], 1.0])
fw_tan = res["slope"] * (Sw_tan - base_case["Swi"])
ax.plot(Sw_tan, fw_tan, "--", color="#c0392b", lw=1.5, label="Tangente de Welge")
ax.plot([res["Swf"]], [res["fwf"]], "o", color="#c0392b", ms=8, label=f"Frente Swf={res['Swf']:.3f}")
ax.axvline(base_case["Swi"], color="gray", lw=0.8, ls=":")
ax.axhline(1.0, color="gray", lw=0.5)
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("Saturación de agua, Sw")
ax.set_ylabel("Flujo fraccional de agua, fw")
ax.set_title("Curva de flujo fraccional y construcción de Welge")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
rc = res["recovery_curve"]
Qi_pre = np.linspace(1e-3, res["Qi_BT"], 50)
RF_pre = Qi_pre  # pre-irrupción: RF_BT * (Qi/Qi_BT) no aplica; ver nota abajo
# Pre-irrupción: por balance de materia, Sw_avg = Swi + Qi (lineal), por lo
# que RF(Qi) = Qi / (1 - Swi) para Qi <= Qi_BT.
RF_pre = Qi_pre / (1.0 - base_case["Swi"])
Qi_plot = np.concatenate((Qi_pre, rc["Qi"]))
RF_plot = np.concatenate((RF_pre, rc["RF"]))
order = np.argsort(Qi_plot)
ax.plot(Qi_plot[order], np.array(RF_plot)[order] * 100, color="#1f4e79", lw=2)
ax.axvline(res["Qi_BT"], color="#c0392b", ls="--", lw=1.2, label=f"Irrupción (Qi={res['Qi_BT']:.2f} PV, RF={res['RF_BT']*100:.1f}%)")
ax.set_xscale("log")
ax.set_xlim(1e-2, 20)
ax.set_xlabel("Volúmenes porosos inyectados, Qi (escala log)")
ax.set_ylabel("Factor de recobro, RF (%)")
ax.set_title("Curva de recobro vs inyección acumulada")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "..", "docs", "validation_plot.png"), dpi=140)
print("\nGráfico guardado en validation_plot.png")
