"""
Modelo de permeabilidades relativas de Corey (Corey, 1954; Brooks & Corey, 1964).

Convención de variables:
    Sw      : saturación de agua (fracción, 0-1)
    Swi     : saturación de agua irreducible (connata)
    Sor     : saturación residual de petróleo
    kro_max : permeabilidad relativa al petróleo en Sw = Swi (end-point)
    krw_max : permeabilidad relativa al agua en Sw = 1 - Sor (end-point)
    no, nw  : exponentes de Corey para petróleo y agua (curvatura de las curvas)

La saturación de agua normalizada (saturación efectiva) es:

    Swn = (Sw - Swi) / (1 - Swi - Sor)        para Swi <= Sw <= 1 - Sor

Y las permeabilidades relativas:

    krw(Sw) = krw_max * Swn ** nw
    kro(Sw) = kro_max * (1 - Swn) ** no
"""

import numpy as np


def normalized_saturation(Sw, Swi, Sor):
    """Saturación de agua normalizada (efectiva), acotada en [0, 1]."""
    Sw = np.asarray(Sw, dtype=float)
    denom = 1.0 - Swi - Sor
    if denom <= 0:
        raise ValueError(
            f"Rango de saturación móvil inválido: 1 - Swi - Sor = {denom:.4f} <= 0 "
            f"(Swi={Swi}, Sor={Sor})"
        )
    Swn = (Sw - Swi) / denom
    return np.clip(Swn, 0.0, 1.0)


def relative_permeabilities(Sw, Swi, Sor, kro_max, krw_max, no, nw):
    """
    Devuelve (kro, krw) evaluados en Sw usando el modelo de Corey.

    Parameters
    ----------
    Sw : float or array
        Saturación de agua a evaluar.
    Swi, Sor : float
        Saturaciones irreducible de agua y residual de petróleo.
    kro_max, krw_max : float
        Permeabilidades relativas extremas (end-points).
    no, nw : float
        Exponentes de Corey para petróleo y agua.

    Returns
    -------
    kro, krw : float or np.ndarray
    """
    Swn = normalized_saturation(Sw, Swi, Sor)
    kro = kro_max * (1.0 - Swn) ** no
    krw = krw_max * (Swn) ** nw
    return kro, krw


def default_corey_ranges():
    """
    Rangos típicos reportados en literatura para areniscas consolidadas
    (usados como referencia para muestreo de datos sintéticos).
    Fuente de criterio general: Willhite (1986); Craig (1971); Ahmed, T.
    Reservoir Engineering Handbook (valores de rango, no cita textual).
    """
    return {
        "no": (1.5, 4.0),
        "nw": (1.5, 4.0),
        "kro_max": (0.6, 0.95),
        "krw_max": (0.15, 0.55),
    }
