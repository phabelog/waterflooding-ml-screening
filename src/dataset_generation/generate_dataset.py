"""
Generación del dataset sintético etiquetado.

Corresponde a los pasos 1.3 y 1.4 del flujo de trabajo metodológico:
"Generación de escenarios sintéticos" y "Cálculo de características físicas
derivadas y etiquetado apto/no apto".

Flujo por escenario:

    1. Muestreo de las variables de entrada dentro de sus rangos.
    2. Aplicación de las relaciones de consistencia física
       (viscosidad condicionada a API y temperatura, saturaciones válidas).
    3. Cálculo de las características derivadas con el framework físico:
         - ED  : Buckley-Leverett + Welge, al corte de agua económico
         - M   : relación de movilidad de punto extremo (Darcy)
         - EA  : barrido areal (Craig / Willhite)
         - EV  : barrido vertical (Dykstra-Parsons)
         - RF_total = ED * EA * EV
         - q_iny : caudal de inyección alcanzable (Darcy radial)
    4. Etiquetado apto / no apto con el criterio de dos niveles.

El archivo resultante contiene tanto las variables de entrada (que verá el
modelo de ML) como las derivadas (que se conservan solo para trazabilidad y
análisis; NO se usan como features de entrenamiento).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "physics"))

from buckley_leverett import displacement_efficiency_at_fw
from darcy import injection_rate_for_dp, endpoint_mobility_ratio
from sweep_efficiency import total_recovery_factor

from variable_ranges import (
    INPUT_VARIABLES, RELPERM_VARIABLES, OPERATIONAL_CONSTANTS,
    oil_viscosity_from_api, water_viscosity_from_temperature,
    current_oil_saturation, is_physically_valid,
)
from labeling import evaluate_scenario


# ----------------------------------------------------------------------
#  Muestreo
# ----------------------------------------------------------------------

def _sample_variable(lo, hi, kind, n, rng):
    if kind == "log":
        return 10.0 ** rng.uniform(np.log10(lo), np.log10(hi), n)
    return rng.uniform(lo, hi, n)


def sample_scenarios(n, seed=42):
    """Muestrea n escenarios válidos y devuelve un DataFrame."""
    rng = np.random.default_rng(seed)
    data = {}

    for name, (lo, hi, kind, _) in INPUT_VARIABLES.items():
        data[name] = _sample_variable(lo, hi, kind, n, rng)
    for name, (lo, hi, kind, _) in RELPERM_VARIABLES.items():
        data[name] = _sample_variable(lo, hi, kind, n, rng)

    df = pd.DataFrame(data)

    # Consistencia física
    df["muo_cp"] = oil_viscosity_from_api(df["api"].values, rng)
    df["muw_cp"] = water_viscosity_from_temperature(df["T_res_F"].values)
    df["So_current"] = current_oil_saturation(
        df["Swi"].values, df["Sor"].values, rng.uniform(0.35, 1.0, n))

    # Descartar combinaciones sin suficiente saturación móvil
    valid = is_physically_valid(df["Swi"].values, df["Sor"].values)
    return df[valid].reset_index(drop=True)


# ----------------------------------------------------------------------
#  Características físicas derivadas
# ----------------------------------------------------------------------

def compute_physics(row, op=None):
    """Ejecuta el framework físico sobre un escenario."""
    op = op or OPERATIONAL_CONSTANTS

    params = dict(
        Swi=row["Swi"], Sor=row["Sor"],
        kro_max=row["kro_max"], krw_max=row["krw_max"],
        no=row["no"], nw=row["nw"],
        muo=row["muo_cp"], muw=row["muw_cp"],
    )

    ED = displacement_efficiency_at_fw(
        params, row["Swi"], row["Sor"], fw_limit=op["fw_limit"])

    M = endpoint_mobility_ratio(
        row["krw_max"], row["muw_cp"], row["kro_max"], row["muo_cp"])

    sweep = total_recovery_factor(
        ED, M, row["Vdp"], wor_limit=op["wor_limit"])

    q_inj = injection_rate_for_dp(
        k_md=row["k_md"], h_ft=row["h_net_ft"], muw_cp=row["muw_cp"],
        re_ft=op["re_ft"], dp_psi=op["dp_max_psi"],
        rw_ft=op["rw_ft"], skin=row["skin"])

    return {
        "ED": ED,
        "M": M,
        "EA": sweep["EA"],
        "EV": sweep["EV"],
        "RF_total": sweep["RF_total"],
        "q_inj_bpd": q_inj,
    }


# ----------------------------------------------------------------------
#  Generación completa
# ----------------------------------------------------------------------

def build_dataset(n=5000, seed=42, verbose=True):
    """Genera el dataset sintético completo y etiquetado."""
    df = sample_scenarios(n, seed)
    if verbose:
        print(f"Escenarios físicamente válidos: {len(df)} de {n}")

    records = []
    for _, row in df.iterrows():
        try:
            phys = compute_physics(row)
        except Exception:
            continue                       # escenario numéricamente inviable

        result = evaluate_scenario(row.to_dict(), phys)
        records.append({**row.to_dict(), **phys,
                        "label": result["label"],
                        "passed_l1": result["passed_l1"],
                        "passed_l2": result["passed_l2"],
                        "reject_reasons": "; ".join(result["reject_reasons"])})

    out = pd.DataFrame(records)
    if verbose and len(out):
        apt = int(out["label"].sum())
        print(f"Dataset final: {len(out)} escenarios")
        print(f"  Aptos    : {apt} ({apt/len(out)*100:.1f} %)")
        print(f"  No aptos : {len(out)-apt} ({(len(out)-apt)/len(out)*100:.1f} %)")
    return out


# Columnas que verá el modelo de ML (NO incluyen las salidas del framework)
FEATURE_COLUMNS = (
    list(INPUT_VARIABLES.keys())
    + list(RELPERM_VARIABLES.keys())
    + ["muo_cp", "muw_cp", "So_current"]
)

# Columnas derivadas: se conservan para trazabilidad, no para entrenar
DERIVED_COLUMNS = ["ED", "M", "EA", "EV", "RF_total", "q_inj_bpd"]


if __name__ == "__main__":
    df = build_dataset(n=5000, seed=42)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "synthetic_dataset.csv")
    df.to_csv(path, index=False)
    print(f"\nGuardado en: {os.path.abspath(path)}")
