"""
Ley de Darcy aplicada a inyección radial de agua, y cálculo de la relación
de movilidad agua-petróleo (end-point mobility ratio, Craig, 1971).

Unidades de campo (field units):
    k   [mD], h [ft], mu [cP], p [psi], q [bbl/d], r [ft]
"""

import numpy as np


def endpoint_mobility_ratio(krw_max, muw, kro_max, muo):
    """
    Relación de movilidad de punto extremo (Craig, 1971):

        M = (krw_max / muw) / (kro_max / muo)

    M < 1  -> desplazamiento favorable (frente estable)
    M > 1  -> desplazamiento desfavorable (riesgo de digitación viscosa /
              canalización de agua)
    """
    lambda_w = krw_max / muw
    lambda_o = kro_max / muo
    return lambda_w / lambda_o


def radial_injectivity_index(k_md, h_ft, muw_cp, re_ft, rw_ft=0.33, skin=0.0):
    """
    Índice de inyectividad radial en estado estacionario, unidades de campo:

        II [bbl/d/psi] = (0.00708 * k * h) / (muw * (ln(re/rw) - 0.75 + S))

    (Forma estándar de la ecuación de Darcy radial para pozos inyectores,
    Craig, 1971; Willhite, 1986). Se usa como indicador de restricción
    operacional de inyectividad, no como sustituto de una prueba de pozo.
    """
    if re_ft <= rw_ft:
        raise ValueError("re_ft debe ser mayor que rw_ft")
    denom = muw_cp * (np.log(re_ft / rw_ft) - 0.75 + skin)
    if denom <= 0:
        raise ValueError("Denominador no positivo en índice de inyectividad; revisar skin/radios.")
    return (0.00708 * k_md * h_ft) / denom


def injection_rate_for_dp(k_md, h_ft, muw_cp, re_ft, dp_psi, rw_ft=0.33, skin=0.0):
    """Caudal de inyección [bbl/d] para un Δp [psi] dado, vía Darcy radial."""
    II = radial_injectivity_index(k_md, h_ft, muw_cp, re_ft, rw_ft, skin)
    return II * dp_psi
