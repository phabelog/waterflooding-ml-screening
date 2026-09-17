"""
Criterio de etiquetado apto / no apto.

Corresponde al paso 1.4 del flujo de trabajo metodológico:
"Cálculo de características físicas derivadas y etiquetado apto/no apto".

El etiquetado se aplica en dos niveles:

  NIVEL 1 — Viabilidad básica (eliminatorio)
      Descarta reservorios donde la inyección de agua no tiene sentido por
      razones elementales, antes de gastar cómputo en el framework físico.

  NIVEL 2 — Desempeño físico (calculado con el framework)
      Evalúa el desempeño esperado del desplazamiento mediante los
      indicadores que produce el framework: recobro total corregido por
      eficiencias de barrido, relación de movilidad, heterogeneidad e
      inyectividad.

Un escenario es APTO únicamente si supera AMBOS niveles.

Los umbrales están centralizados en el diccionario THRESHOLDS para que
puedan revisarse y ajustarse sin tocar la lógica.
"""

import numpy as np

from variable_ranges import OPERATIONAL_CONSTANTS


# ======================================================================
#  UMBRALES DEL CRIBADO
# ======================================================================

THRESHOLDS = {
    # ---- Nivel 1: viabilidad básica ----
    "k_min_md": 10.0,
    # Por debajo de ~10 mD la inyectividad no sostiene caudales de campo.

    "h_net_min_ft": 10.0,
    # Espesor neto mínimo para que el volumen barrido justifique la inversión.

    "So_min": 0.45,
    # El método de Dykstra-Parsons se formuló para saturaciones de petróleo
    # del 45 % o mayores. El umbral se fija EN ese valor, no por debajo: un
    # umbral menor (por ejemplo 0.40) sería más PERMISIVO, no más
    # conservador, porque admitiría escenarios fuera del rango en que el
    # método de barrido vertical fue validado. Mantenerlo en 0.45 asegura
    # que todo caso etiquetado como apto haya sido evaluado dentro del
    # dominio de validez de las correlaciones empleadas.

    "muo_max_cp": 50.0,
    # Por encima de este valor el frente es inestable y el caso corresponde
    # a inyección de polímero, no a waterflooding convencional.

    # ---- Nivel 2: desempeño físico ----
    "M_max": 10.0,
    # Filtro de casos patológicos, no criterio fino. El efecto de la
    # movilidad sobre el desempeño YA está incorporado en RF_total a través
    # de las eficiencias de barrido areal EA(M) y vertical EV(M, Vdp), por
    # lo que un umbral estricto adicional sobre M duplicaría el mismo
    # criterio. Se corta en 10 porque es el límite superior del rango donde
    # la correlación de barrido areal fue ajustada (EA ~ 50 % en M = 10).

    "Vdp_max": 0.80,
    # La mayoría de reservorios cae entre 0.5 y 0.9; por encima de 0.80 la
    # canalización por las capas de alta permeabilidad degrada el barrido
    # vertical de forma severa.

    "RF_total_min": 0.15,
    # Recobro incremental mínimo (fracción del OOIP) para que el proyecto
    # tenga sentido técnico-económico, ya corregido por barrido areal y
    # vertical.

    "injectivity_min_bpd": 150.0,
    # Caudal de inyección mínimo por pozo [bbl/d] alcanzable con la caída de
    # presión disponible.
}


# ======================================================================
#  EVALUACIÓN
# ======================================================================

def level_1_basic_viability(k_md, h_net_ft, So_current, muo_cp, th=None):
    """
    Nivel 1: viabilidad básica. Devuelve (pasa, motivos_de_rechazo).
    """
    th = th or THRESHOLDS
    reasons = []

    if k_md < th["k_min_md"]:
        reasons.append(f"k={k_md:.1f} mD < {th['k_min_md']:.0f} mD")
    if h_net_ft < th["h_net_min_ft"]:
        reasons.append(f"h={h_net_ft:.1f} ft < {th['h_net_min_ft']:.0f} ft")
    if So_current < th["So_min"]:
        reasons.append(f"So={So_current:.2f} < {th['So_min']:.2f}")
    if muo_cp > th["muo_max_cp"]:
        reasons.append(f"muo={muo_cp:.1f} cP > {th['muo_max_cp']:.0f} cP")

    return len(reasons) == 0, reasons


def level_2_physical_performance(M, Vdp, RF_total, q_inj_bpd, th=None):
    """
    Nivel 2: desempeño físico. Devuelve (pasa, motivos_de_rechazo).
    """
    th = th or THRESHOLDS
    reasons = []

    if M > th["M_max"]:
        reasons.append(f"M={M:.2f} > {th['M_max']:.1f}")
    if Vdp > th["Vdp_max"]:
        reasons.append(f"Vdp={Vdp:.2f} > {th['Vdp_max']:.2f}")
    if RF_total < th["RF_total_min"]:
        reasons.append(f"RF={RF_total*100:.1f}% < {th['RF_total_min']*100:.0f}%")
    if q_inj_bpd < th["injectivity_min_bpd"]:
        reasons.append(f"q_iny={q_inj_bpd:.0f} bbl/d < {th['injectivity_min_bpd']:.0f} bbl/d")

    return len(reasons) == 0, reasons


def evaluate_scenario(scenario, physics, th=None):
    """
    Aplica los dos niveles y devuelve la etiqueta final.

    Parameters
    ----------
    scenario : dict
        Propiedades del reservorio (k_md, h_net_ft, So_current, muo_cp, Vdp).
    physics : dict
        Salidas del framework físico (M, RF_total, q_inj_bpd).

    Returns
    -------
    dict con:
        label        : 1 = apto, 0 = no apto
        passed_l1    : bool
        passed_l2    : bool
        reject_reasons : lista de motivos (vacía si es apto)
    """
    th = th or THRESHOLDS

    ok1, r1 = level_1_basic_viability(
        scenario["k_md"], scenario["h_net_ft"],
        scenario["So_current"], scenario["muo_cp"], th)

    # El nivel 2 se evalúa siempre, para poder analizar después qué criterio
    # resultó determinante en cada rechazo.
    ok2, r2 = level_2_physical_performance(
        physics["M"], scenario["Vdp"],
        physics["RF_total"], physics["q_inj_bpd"], th)

    return {
        "label": int(ok1 and ok2),
        "passed_l1": bool(ok1),
        "passed_l2": bool(ok2),
        "reject_reasons": r1 + r2,
    }


def thresholds_table(th=None):
    """Tabla de texto con los umbrales vigentes, para documentación."""
    th = th or THRESHOLDS
    rows = [
        ("NIVEL 1 — Viabilidad básica", "", ""),
        ("Permeabilidad",            f">= {th['k_min_md']:.0f}", "mD"),
        ("Espesor neto",             f">= {th['h_net_min_ft']:.0f}", "ft"),
        ("Saturación de petróleo",   f">= {th['So_min']:.2f}", "fracción"),
        ("Viscosidad del petróleo",  f"<= {th['muo_max_cp']:.0f}", "cP"),
        ("", "", ""),
        ("NIVEL 2 — Desempeño físico", "", ""),
        ("Relación de movilidad",    f"<= {th['M_max']:.1f}", "adimensional"),
        ("Coef. Dykstra-Parsons",    f"<= {th['Vdp_max']:.2f}", "adimensional"),
        ("Recobro total",            f">= {th['RF_total_min']*100:.0f}", "% OOIP"),
        ("Caudal de inyección",      f">= {th['injectivity_min_bpd']:.0f}", "bbl/d"),
    ]
    lines = [f"{'Criterio':<28}{'Umbral':>12}  {'Unidad'}", "-" * 58]
    for name, val, unit in rows:
        if not name:
            lines.append("")
        elif not val:
            lines.append(name)
        else:
            lines.append(f"{name:<28}{val:>12}  {unit}")
    return "\n".join(lines)
