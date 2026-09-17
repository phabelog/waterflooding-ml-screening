"""
Teoría de desplazamiento de Buckley-Leverett (Buckley & Leverett, 1942) con
construcción gráfica de Welge (Welge, 1952) para determinar el frente de
choque (shock front), la saturación promedio de agua y el factor de recobro.

Se ignoran presión capilar y efectos gravitacionales (supuesto estándar para
un flujo de trabajo de screening 1-D; ver Willhite, 1986). El objetivo NO es
sustituir un simulador de yacimientos, sino producir indicadores físicamente
consistentes (RF, tiempo de irrupción, relación de movilidad) que alimenten
el dataset sintético para el modelo de ML.
"""

import numpy as np
from scipy.optimize import brentq

from corey import relative_permeabilities


def fractional_flow(Sw, Swi, Sor, kro_max, krw_max, no, nw, muo, muw):
    """
    Flujo fraccional de agua fw(Sw), sin gravedad ni presión capilar:

        fw = 1 / (1 + (kro * muw) / (krw * muo))

    En Sw = Swi -> krw = 0 -> fw = 0
    En Sw = 1 - Sor -> kro = 0 -> fw = 1
    """
    kro, krw = relative_permeabilities(Sw, Swi, Sor, kro_max, krw_max, no, nw)
    # Evitar división por cero cuando krw = 0 (Sw = Swi)
    with np.errstate(divide="ignore", invalid="ignore"):
        mobility_ratio_local = np.where(krw > 0, (kro * muw) / (krw * muo + 1e-30), np.inf)
        fw = 1.0 / (1.0 + mobility_ratio_local)
    fw = np.nan_to_num(fw, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(fw, 0.0, 1.0)


def _fw_of_Sw(Sw, params):
    return fractional_flow(Sw, **params)


def find_shock_front(params, Swi, Sor, n_grid=20000, Sw_start=None):
    """
    Determina la saturación de frente de choque (Swf) mediante la
    construcción de Welge: la tangente trazada desde el punto inicial
    (Sw_start, fw(Sw_start)) hasta la curva fw(Sw).

    Sw_start es la saturación de agua existente al inicio de la inyección.
    Si el reservorio no ha producido por energía primaria, coincide con la
    saturación de agua irreducible Swi y la tangente parte de (Swi, 0), que
    es el caso clásico. Si ya hubo producción primaria, parte del agua
    móvil está presente antes de inyectar y la tangente debe trazarse desde
    ese punto: hacerlo desde Swi sobrestimaría el petróleo recuperable.

    Se valida con DOS métodos numéricos independientes:
      (A) Búsqueda en malla: maximizar la pendiente secante fw(Sw)/(Sw-Swi).
      (B) Búsqueda de raíz: resolver la condición de tangencia
          fw'(Sw) - fw(Sw)/(Sw-Swi) = 0 con Brent.

    Ambos métodos deben converger al mismo punto (Swf, fwf); si difieren
    en más de la tolerancia, se lanza un error (indicaría una curva fw
    no convexa/cóncava simple, fuera del alcance de este framework).

    Returns
    -------
    dict con Swf, fwf, slope (= fw'(Swf) = pendiente de la tangente)
    """
    if Sw_start is None:
        Sw_start = Swi
    if not (Swi - 1e-9 <= Sw_start < 1 - Sor):
        raise ValueError(
            f"Sw_start={Sw_start:.4f} debe estar en [Swi, 1-Sor) "
            f"(Swi={Swi}, Sor={Sor})")

    fw_start = float(_fw_of_Sw(Sw_start, params))

    Sw_grid = np.linspace(Sw_start + 1e-6, 1 - Sor - 1e-6, n_grid)
    fw_grid = _fw_of_Sw(Sw_grid, params)

    # --- Método A: máxima pendiente secante desde (Sw_start, fw_start) ---
    secant_slope = (fw_grid - fw_start) / (Sw_grid - Sw_start)
    idx_max = int(np.argmax(secant_slope))
    Swf_A = Sw_grid[idx_max]
    fwf_A = fw_grid[idx_max]
    slope_A = secant_slope[idx_max]

    # --- Método B: raíz de la condición de tangencia (derivada numérica) ---
    def tangency_gap(Sw):
        h = 1e-5
        fw_p = _fw_of_Sw(min(Sw + h, 1 - Sor - 1e-7), params)
        fw_m = _fw_of_Sw(max(Sw - h, Swi + 1e-7), params)
        dfw_dSw = (fw_p - fw_m) / (2 * h)
        fw_here = _fw_of_Sw(Sw, params)
        secant = (fw_here - fw_start) / (Sw - Sw_start)
        return dfw_dSw - secant

    lo, hi = Sw_start + 1e-4, 1 - Sor - 1e-4
    # Buscar cambio de signo cerca del óptimo de la malla para acotar Brent
    window = max(50 * (Sw_grid[1] - Sw_grid[0]), 1e-3)
    lo_b = max(lo, Swf_A - window)
    hi_b = min(hi, Swf_A + window)
    try:
        Swf_B = brentq(tangency_gap, lo_b, hi_b, xtol=1e-8)
    except ValueError:
        # Si no hay cambio de signo en la ventana, se confía en el método A
        Swf_B = Swf_A
    fwf_B = _fw_of_Sw(Swf_B, params)
    slope_B = (fwf_B - fw_start) / (Swf_B - Sw_start)

    diff = abs(Swf_A - Swf_B)
    if diff > 5e-3:
        raise RuntimeError(
            f"Los dos métodos de localización del frente de choque no "
            f"coinciden (Swf_A={Swf_A:.4f} vs Swf_B={Swf_B:.4f}, "
            f"diff={diff:.4f}). Revisar forma de la curva fw(Sw)."
        )

    return {
        "Swf": float(Swf_B),
        "fwf": float(fwf_B),
        "slope": float(slope_B),
        "Sw_start": float(Sw_start),
        "fw_start": float(fw_start),
        "cross_check_diff": float(diff),
        "Sw_grid": Sw_grid,
        "fw_grid": fw_grid,
    }


def welge_results(params, Swi, Sor, Sw_start=None):
    """
    Calcula los resultados clásicos de Welge en el punto de irrupción
    (breakthrough) y la curva de recobro post-irrupción.

    Returns
    -------
    dict con:
      Swf, fwf, slope           : frente de choque
      Qi_BT                     : volúmenes porosos inyectados a la irrupción
      Sw_avg_BT                 : saturación promedio de agua a la irrupción
      RF_BT                     : factor de recobro (fracción de OOIP) a la irrupción
      recovery_curve            : (Qi[], RF[]) desde irrupción hasta Sw=1-Sor
    """
    shock = find_shock_front(params, Swi, Sor, Sw_start=Sw_start)
    Swf, fwf, slope = shock["Swf"], shock["fwf"], shock["slope"]
    Sw_start = shock["Sw_start"]
    fw_start = shock["fw_start"]

    Qi_BT = 1.0 / slope
    # Sw_avg a la irrupción: balance de materia (el agua inyectada que no ha
    # salido permanece en el yacimiento) == construcción de Welge (extensión
    # de la tangente hasta fw = 1). Ambas expresiones deben coincidir; se usa
    # como verificación cruzada adicional.
    Sw_avg_BT_material_balance = Sw_start + (1.0 - fw_start) * Qi_BT
    Sw_avg_BT_welge = Sw_start + (1.0 - fw_start) / slope
    if abs(Sw_avg_BT_material_balance - Sw_avg_BT_welge) > 1e-9:
        raise RuntimeError("Inconsistencia entre balance de materia y construcción de Welge.")
    Sw_avg_BT = Sw_avg_BT_welge

    # Recobro referido al petróleo presente al INICIO de la inyección
    RF_BT = (Sw_avg_BT - Sw_start) / (1.0 - Sw_start)

    # --- Curva de recobro post-irrupción (familia de tangentes de Welge) ---
    Sw2_array = np.linspace(Swf, 1 - Sor - 1e-6, 200)
    fw2_array = _fw_of_Sw(Sw2_array, params)
    h = 1e-5
    dfw2 = (
        _fw_of_Sw(np.minimum(Sw2_array + h, 1 - Sor - 1e-7), params)
        - _fw_of_Sw(np.maximum(Sw2_array - h, Swi + 1e-7), params)
    ) / (2 * h)
    dfw2 = np.clip(dfw2, 1e-8, None)

    Qi_array = 1.0 / dfw2
    Sw_avg_array = Sw2_array + (1.0 - fw2_array) / dfw2
    RF_array = (Sw_avg_array - Sw_start) / (1.0 - Sw_start)
    RF_array = np.clip(RF_array, 0.0, 1.0)

    return {
        "Swf": Swf,
        "fwf": fwf,
        "slope": slope,
        "Qi_BT": Qi_BT,
        "Sw_avg_BT": Sw_avg_BT,
        "RF_BT": RF_BT,
        "Sw_start": Sw_start,
        "fw_start": fw_start,
        "cross_check_diff": shock["cross_check_diff"],
        "recovery_curve": {"Qi": Qi_array, "RF": RF_array, "Sw2": Sw2_array},
        "fw_curve": {"Sw": shock["Sw_grid"], "fw": shock["fw_grid"]},
    }


def displacement_at_economic_limit(params, Swi, Sor, fw_limit, Sw_start=None):
    """
    Estado del desplazamiento cuando el corte de agua producido alcanza el
    límite económico fw_limit.

    A diferencia de la versión anterior, devuelve además los volúmenes
    porosos inyectados a la irrupción y al límite económico. Esa relación es
    necesaria para evaluar el barrido areal en el MISMO instante que la
    eficiencia de desplazamiento, en lugar de mezclar estados distintos.

    Se localiza la saturación Sw2 en la cara productora donde fw(Sw2) =
    fw_limit y se aplica la construcción de Welge:

        Sw_avg = Sw2 + (1 - fw2) / (dfw/dSw)|_Sw2
        ED     = (Sw_avg - Sw_start) / (1 - Sw_start)
        Qi     = 1 / (dfw/dSw)|_Sw2

    ED se refiere al petróleo presente al inicio de la inyección, no al
    petróleo original en sitio: el waterflooding solo puede recuperar lo que
    queda tras la producción primaria.

    Returns
    -------
    dict con ED, Qi_BT, Qi_econ, Swf, Sw_start
    """
    res = welge_results(params, Swi, Sor, Sw_start=Sw_start)
    Sw_start = res["Sw_start"]

    # Si la irrupción ocurre ya por encima del corte límite, el límite
    # económico se alcanza en el instante mismo de la irrupción.
    if fw_limit <= res["fwf"]:
        return {"ED": float(np.clip(res["RF_BT"], 0.0, 1.0)),
                "Qi_BT": float(res["Qi_BT"]), "Qi_econ": float(res["Qi_BT"]),
                "Swf": float(res["Swf"]), "Sw_start": float(Sw_start)}

    Sw_lo, Sw_hi = res["Swf"], 1 - Sor - 1e-7

    def gap(Sw):
        return float(_fw_of_Sw(Sw, params)) - fw_limit

    Sw2 = Sw_hi if gap(Sw_hi) < 0 else brentq(gap, Sw_lo, Sw_hi, xtol=1e-9)

    h = 1e-5
    fw2 = float(_fw_of_Sw(Sw2, params))
    dfw = (float(_fw_of_Sw(min(Sw2 + h, 1 - Sor - 1e-7), params))
           - float(_fw_of_Sw(max(Sw2 - h, Sw_start + 1e-7), params))) / (2 * h)
    dfw = max(dfw, 1e-8)

    Sw_avg = Sw2 + (1.0 - fw2) / dfw
    ED = (Sw_avg - Sw_start) / (1.0 - Sw_start)

    return {"ED": float(np.clip(ED, 0.0, 1.0)),
            "Qi_BT": float(res["Qi_BT"]), "Qi_econ": float(1.0 / dfw),
            "Swf": float(res["Swf"]), "Sw_start": float(Sw_start)}
