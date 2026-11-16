# -*- coding: utf-8 -*-
"""
Sensibilidad del etiquetado a sus supuestos (Sección 3.7 del artículo y
Tablas C2 y C3 de docs/Informacion_complementaria).

Recalcula el criterio físico sobre los 5 000 escenarios sintéticos variando
UN supuesto a la vez y registra la fracción apta y el porcentaje de etiquetas
que cambian respecto del caso base:

  1. WOR de corte: 10, 25 (base) y 50, con fw = WOR / (1 + WOR).
  2. Cada uno de los ocho umbrales de la Tabla 2 del artículo, en un valor más
     permisivo y uno más exigente.
  3. Caso de referencia (2 000 realizaciones): efecto de los supuestos de
     imputación (saturación actual de petróleo y viscosidad inferida).

Uso (desde la raíz del repositorio):
    python src/evaluation/sensitivity.py

Resultado: results/sensibilidad_etiquetado.json (tarda alrededor de un minuto).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for sub in ("src/physics", "src/dataset_generation", "src/preprocessing",
            "src/models", "src/evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from generate_dataset import compute_physics          # noqa: E402
from variable_ranges import OPERATIONAL_CONSTANTS     # noqa: E402
from labeling import evaluate_scenario, THRESHOLDS    # noqa: E402
import evaluate as ev                                 # noqa: E402

DATASET = os.path.join(ROOT, "data", "synthetic", "synthetic_dataset.csv")
OUT = os.path.join(ROOT, "results", "sensibilidad_etiquetado.json")

# Valores probados para cada umbral: (más permisivo, más exigente)
THRESHOLD_TESTS = {
    "k_min_md": (5.0, 20.0),
    "h_net_min_ft": (5.0, 20.0),
    "So_min": (0.40, 0.50),
    "muo_max_cp": (100.0, 30.0),
    "M_max": (15.0, 5.0),
    "Vdp_max": (0.90, 0.70),
    "RF_total_min": (0.10, 0.20),
    "injectivity_min_bpd": (100.0, 300.0),
}


def label_and_physics(df, op=None, th=None):
    labels, physics = [], []
    for _, row in df.iterrows():
        ph = compute_physics(row, op) if op is not None else compute_physics(row)
        labels.append(evaluate_scenario(row.to_dict(), ph, th)["label"])
        physics.append(ph)
    return np.array(labels), pd.DataFrame(physics)


def wor_sensitivity(df):
    base = df["label"].values
    out = {}
    for wor in (10.0, 25.0, 50.0):
        op = dict(OPERATIONAL_CONSTANTS)
        op["wor_limit"] = wor
        op["fw_limit"] = wor / (1.0 + wor)
        lab, ph = label_and_physics(df, op=op)
        out[f"WOR_{int(wor)}"] = {
            "fw": round(wor / (1.0 + wor), 4),
            "RF_medio_pct": round(100 * ph["RF_total"].mean(), 1),
            "RF_p50_pct": round(100 * ph["RF_total"].median(), 1),
            "aptos_pct": round(100 * lab.mean(), 1),
            "etiquetas_que_cambian_pct": round(100 * (lab != base).mean(), 1),
        }
    return out


def threshold_sensitivity(df):
    base = df["label"].values
    out = {}
    for key, values in THRESHOLD_TESTS.items():
        rows = []
        for v in values:
            th = dict(THRESHOLDS)
            th[key] = v
            # Las magnitudes físicas no dependen de los umbrales: se reutilizan
            # las columnas derivadas guardadas en el dataset.
            lab = np.array([evaluate_scenario(r.to_dict(), r.to_dict(), th)["label"]
                            for _, r in df.iterrows()])
            rows.append({"valor": v,
                         "aptos_pct": round(100 * lab.mean(), 1),
                         "etiquetas_que_cambian_pct": round(100 * (lab != base).mean(), 1)})
        out[key] = {"base": THRESHOLDS[key], "pruebas": rows}
    return out


def reference_case_sensitivity(n=2000):
    base = ev.reference_case_scenarios(n)

    def run(d):
        lab, ph = label_and_physics(d)
        return {"aptos_fisicos_pct": round(100 * lab.mean(), 1),
                "M_p50": round(float(ph["M"].median()), 1),
                "RF_p50_pct": round(100 * float(ph["RF_total"].median()), 1)}

    out = {"base (So,act = 1 - Swi)": run(base)}
    for frac in (0.10, 0.20):
        d = base.copy()
        d["So_current"] = 1 - d["Swi"] - frac * (1 - d["Swi"] - d["Sor"])
        out[f"{int(frac * 100)} % del petróleo móvil ya producido"] = run(d)
    for fac in (2.0, 0.5):
        d = base.copy()
        d["muo_cp"] = d["muo_cp"] * fac
        out[f"viscosidad del petróleo x {fac}"] = run(d)
    return out


def main():
    df = pd.read_csv(DATASET)
    result = {
        "wor_de_corte": wor_sensitivity(df),
        "umbrales": threshold_sensitivity(df),
        "caso_de_referencia": reference_case_sensitivity(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
