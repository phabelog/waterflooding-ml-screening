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


def radial_injectivity_index(k_md, h_ft, muw_cp, re_ft, krw=1.0, Bw=1.02,
                             rw_ft=0.33, skin=0.0):
    """
    Índice de inyectividad radial en régimen PSEUDOESTACIONARIO, unidades de
    campo:

        II [bbl/d/psi] = 0.00708 · k · krw · h
                         ---------------------------------
                         μw · Bw · [ ln(re/rw) − 0.75 + S ]

    Precisiones sobre la formulación:

    1) El término −0.75 corresponde al régimen pseudoestacionario, referido
       a la presión promedio del área de drenaje. La forma estrictamente
       estacionaria (presión constante en el radio externo) no lleva ese
       término. Se adopta la pseudoestacionaria por ser la representativa de
       un patrón de inyección confinado.

    2) krw es la permeabilidad relativa al agua en la zona barrida detrás
       del frente. La inyección de agua no ocurre a permeabilidad absoluta:
       en presencia de petróleo residual, la permeabilidad efectiva al agua
       es una fracción de la absoluta. Usar k en lugar de k·krw sobrestima
       la inyectividad por un factor de entre dos y seis, según la roca.
       Nota: se emplea un valor único de krw (el de punto extremo), lo que
       no captura la evolución de la inyectividad con la saturación durante
       el avance del frente; es una simplificación coherente con el nivel de
       detalle de un cribado técnico.

    3) Bw es el factor volumétrico del agua de inyección. Con él, el caudal
       queda expresado en barriles de SUPERFICIE por día; sin él quedaría en
       barriles de reservorio.

    Referencias: Craig (1971); Willhite (1986).
    """
    if re_ft <= rw_ft:
        raise ValueError("re_ft debe ser mayor que rw_ft")
    if not (0.0 < krw <= 1.0):
        raise ValueError(f"krw debe estar en (0, 1]; se recibió {krw}")
    if Bw <= 0:
        raise ValueError("Bw debe ser positivo")

    denom = muw_cp * Bw * (np.log(re_ft / rw_ft) - 0.75 + skin)
    if denom <= 0:
        raise ValueError(
            "Denominador no positivo en el índice de inyectividad; "
            "revisar skin y radios.")
    return (0.00708 * k_md * krw * h_ft) / denom


def injection_rate_for_dp(k_md, h_ft, muw_cp, re_ft, dp_psi, krw=1.0,
                          Bw=1.02, rw_ft=0.33, skin=0.0):
    """Caudal de inyección [bbl/d de superficie] para un Δp [psi] dado."""
    II = radial_injectivity_index(k_md, h_ft, muw_cp, re_ft, krw, Bw,
                                  rw_ft, skin)
    return II * dp_psi
