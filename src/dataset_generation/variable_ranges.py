"""
Definición de las variables de entrada y sus rangos de muestreo.

Corresponde al paso 1.2 del flujo de trabajo metodológico:
"Definición de variables de entrada y sus rangos".

Los rangos se orientan a reservorios de arenisca de la Cuenca Oriente
(Activo Auca: Basal Tena, Napo "U", Napo "T" y Hollín), que es el entorno
donde se hará la validación externa con datos de EP Petroecuador. Se
mantienen deliberadamente más amplios que los valores promedio reportados
para el campo, de modo que el dataset cubra también escenarios claramente
no aptos y el modelo aprenda la frontera de decisión, no solo la zona
favorable.

IMPORTANTE — separación entre entradas y etiqueta
-------------------------------------------------
Las variables de este módulo son las ÚNICAS que verá el modelo de Machine
Learning. Las salidas del framework físico (ED, EA, EV, RF_total, M,
inyectividad) se usan EXCLUSIVAMENTE para construir la etiqueta apto /
no apto. Si se entrenara con esas salidas como entradas, el modelo
reproduciría de forma trivial la regla de etiquetado en vez de aprender la
relación entre las propiedades del reservorio y su aptitud.
"""

import numpy as np


# ----------------------------------------------------------------------
#  Variables de entrada del modelo de ML (features)
# ----------------------------------------------------------------------
# nombre : (mínimo, máximo, tipo de muestreo, descripción)
INPUT_VARIABLES = {
    "k_md":        (5.0,    3000.0, "log",     "Permeabilidad absoluta [mD]"),
    "phi":         (0.08,   0.25,   "uniform", "Porosidad [fracción]"),
    "h_net_ft":    (5.0,    80.0,   "uniform", "Espesor neto saturado [ft]"),
    "api":         (14.0,   35.0,   "uniform", "Gravedad API del petróleo"),
    "Swi":         (0.10,   0.45,   "uniform", "Saturación de agua irreducible"),
    "Sor":         (0.15,   0.35,   "uniform", "Saturación residual de petróleo"),
    "Vdp":         (0.30,   0.90,   "uniform", "Coeficiente de Dykstra-Parsons"),
    "depth_ft":    (7000.0, 10500.0,"uniform", "Profundidad del reservorio [ft]"),
    "T_res_F":     (180.0,  230.0,  "uniform", "Temperatura del reservorio [°F]"),
    "skin":        (-2.0,   8.0,    "uniform", "Factor de daño del pozo inyector"),
}

# Variables que describen las curvas de permeabilidad relativa. Son entradas
# válidas del modelo, pero en la práctica muchas veces no se dispone de
# análisis SCAL; por eso se manejan en un grupo aparte y el estudio evaluará
# el desempeño del modelo con y sin ellas.
RELPERM_VARIABLES = {
    "kro_max": (0.60, 0.95, "uniform", "kro en Sw = Swi (end-point)"),
    "krw_max": (0.15, 0.55, "uniform", "krw en Sw = 1 - Sor (end-point)"),
    "no":      (1.5,  4.0,  "uniform", "Exponente de Corey - petróleo"),
    "nw":      (1.5,  4.0,  "uniform", "Exponente de Corey - agua"),
}

# Variables derivadas de forma condicionada (no se muestrean de forma
# independiente para no generar combinaciones físicamente imposibles).
CONDITIONED_VARIABLES = {
    "muo_cp":      "Viscosidad del petróleo, condicionada a la gravedad API",
    "muw_cp":      "Viscosidad del agua, condicionada a la temperatura",
    "So_current":  "Saturación actual de petróleo, entre Sor y (1 - Swi)",
}

# Constantes operacionales del escenario de inyección
OPERATIONAL_CONSTANTS = {
    "re_ft": 1000.0,      # radio de drenaje del patrón
    "rw_ft": 0.33,        # radio del pozo
    "dp_max_psi": 1000.0, # caída de presión disponible para inyección
    "wor_limit": 25.0,    # WOR económico límite [bbl agua / bbl petróleo]
    "fw_limit": 0.96,     # corte de agua equivalente al WOR límite
}


# ----------------------------------------------------------------------
#  Relaciones de consistencia física
# ----------------------------------------------------------------------

def oil_viscosity_from_api(api, rng, scatter_log10=0.25):
    """
    Viscosidad del petróleo [cP] condicionada a la gravedad API.

    Muestrear API y viscosidad de forma independiente generaría escenarios
    imposibles (crudo de 32 °API con 80 cP, por ejemplo). Se impone una
    tendencia decreciente en escala logarítmica con dispersión aleatoria:

        log10(muo_mediana) = 2.0 - 0.06 * API

    IMPORTANTE: el valor representa la viscosidad del CRUDO VIVO a
    condiciones de reservorio (con gas en solución), no la del crudo muerto
    en superficie, que es varias veces mayor. La expresión está ajustada
    para dar del orden de 1-3 cP entre 28 y 32 °API y algunas decenas de cP
    por debajo de 16 °API, que es el rango habitual a condiciones de
    yacimiento en la Cuenca Oriente.

    Esto NO pretende ser una correlación PVT publicada: es una restricción
    de consistencia interna del dataset sintético. Los valores reales se
    contrastarán contra los PVT que entregue EP Petroecuador.
    """
    api = np.asarray(api, dtype=float)
    log_med = 2.0 - 0.06 * api
    noise = rng.normal(0.0, scatter_log10, size=api.shape)
    muo = 10.0 ** (log_med + noise)
    return np.clip(muo, 0.3, 200.0)


def water_viscosity_from_temperature(T_F):
    """
    Viscosidad del agua [cP] en función de la temperatura del reservorio.

    Se usa una forma sencilla de tipo exponencial decreciente, calibrada
    para dar aprox. 1.0 cP a 60 °F y aprox. 0.30 cP a 210 °F, que es el
    comportamiento estándar del agua de formación en el rango de interés.
    """
    T_F = np.asarray(T_F, dtype=float)
    muw = 1.0 * np.exp(-0.008 * (T_F - 60.0))
    return np.clip(muw, 0.20, 1.10)


def current_oil_saturation(Swi, Sor, fraction):
    """
    Saturación actual de petróleo, garantizando Sor <= So <= 1 - Swi.

    'fraction' en [0, 1] indica qué parte del petróleo móvil permanece en
    el reservorio al momento de evaluar la inyección (1 = no se ha producido
    nada por energía primaria; 0 = solo queda el residual).
    """
    Swi = np.asarray(Swi, dtype=float)
    Sor = np.asarray(Sor, dtype=float)
    fraction = np.asarray(fraction, dtype=float)
    movable = np.clip(1.0 - Swi - Sor, 0.0, None)
    return Sor + fraction * movable


def is_physically_valid(Swi, Sor, min_movable=0.10):
    """
    Un escenario es físicamente válido si queda suficiente saturación móvil:

        1 - Swi - Sor >= min_movable
    """
    return (1.0 - np.asarray(Swi) - np.asarray(Sor)) >= min_movable


def summary_table():
    """Devuelve una tabla de texto con todas las variables y sus rangos."""
    lines = []
    lines.append(f"{'Variable':<14}{'Mín':>10}{'Máx':>12}  {'Muestreo':<10}Descripción")
    lines.append("-" * 92)
    for group, title in [(INPUT_VARIABLES, "Propiedades del reservorio"),
                         (RELPERM_VARIABLES, "Curvas de permeabilidad relativa")]:
        lines.append(f"[{title}]")
        for name, (lo, hi, kind, desc) in group.items():
            lines.append(f"{name:<14}{lo:>10.2f}{hi:>12.2f}  {kind:<10}{desc}")
        lines.append("")
    lines.append("[Variables condicionadas]")
    for name, desc in CONDITIONED_VARIABLES.items():
        lines.append(f"{name:<14}{'—':>10}{'—':>12}  {'derivada':<10}{desc}")
    return "\n".join(lines)
