"""
Eficiencias de barrido areal y vertical.

El recobro real de un waterflooding no es únicamente la eficiencia de
desplazamiento 1-D que entrega Buckley-Leverett: el agua tampoco barre toda
el área del patrón ni todo el espesor del reservorio. El recobro total se
expresa como el producto de las tres eficiencias:

    RF_total = ED * EA * EV

    ED : eficiencia de desplazamiento  (Buckley-Leverett / Welge)
    EA : eficiencia de barrido areal   (patrón de inyección y M)
    EV : eficiencia de barrido vertical (heterogeneidad y M)

--------------------------------------------------------------------------
EA — Barrido areal
--------------------------------------------------------------------------
Craig (1971) presentó la correlación gráfica de EA a la irrupción en función
de la relación de movilidad para un patrón de cinco puntos (five-spot).
Willhite (1986) ajustó esa curva a la forma matemática usada aquí.

--------------------------------------------------------------------------
EV — Barrido vertical
--------------------------------------------------------------------------
Se implementa el modelo estratificado de Dykstra & Parsons (1950)
resolviendo directamente las ecuaciones de avance del frente, en lugar de
ajustar las cartas gráficas de Johnson (1956). Supuestos del modelo:

  - capas horizontales continuas de igual espesor y porosidad,
  - sin flujo cruzado entre capas,
  - desplazamiento tipo pistón dentro de cada capa,
  - misma caída de presión total sobre todas las capas.

La permeabilidad se distribuye log-normal, que es la base de la definición
del coeficiente de Dykstra-Parsons:

    Vdp = (k50 - k84.1) / k50   ==>   sigma_ln = -ln(1 - Vdp)

Referencias:
    Craig, F. F. (1971). The Reservoir Engineering Aspects of Waterflooding.
    Willhite, G. P. (1986). Waterflooding.
    Dykstra, H. & Parsons, R. L. (1950). The prediction of oil recovery by
        water flood.
    Johnson, C. E. (1956). JPT 8(11), 55-56.
"""

import numpy as np
from scipy.stats import norm


# ======================================================================
#  EFICIENCIA DE BARRIDO AREAL
# ======================================================================

def areal_sweep_breakthrough(M):
    """
    Eficiencia de barrido areal a la irrupción, patrón de cinco puntos.

    Correlación de Willhite (1986) sobre la curva gráfica de Craig (1971):

        EA_BT = 0.54602036 + 0.03170817/M + 0.30222997/exp(M) - 0.00509693*M

    Parameters
    ----------
    M : float or array
        Relación de movilidad de punto extremo (adimensional, > 0).

    Returns
    -------
    EA_BT : float or np.ndarray
        Eficiencia de barrido areal, acotada al rango físico (0, 1].

    Notas
    -----
    La correlación fue ajustada en el rango aproximado 0.1 <= M <= 10.
    Fuera de ese rango se extrapola y se acota; por eso el framework
    limita el resultado a [0.25, 1.0], valores coherentes con el
    comportamiento reportado (EA cae de ~100 % a ~50 % cuando M pasa de
    0.15 a 10).
    """
    M = np.asarray(M, dtype=float)
    if np.any(M <= 0):
        raise ValueError("La relación de movilidad M debe ser positiva.")

    EA = (0.54602036
          + 0.03170817 / M
          + 0.30222997 / np.exp(M)
          - 0.00509693 * M)

    return np.clip(EA, 0.25, 1.0)


# ======================================================================
#  EFICIENCIA DE BARRIDO VERTICAL (DYKSTRA-PARSONS)
# ======================================================================

def _layer_permeabilities(Vdp, n_layers, k_median=1.0):
    """
    Genera n_layers permeabilidades log-normales consistentes con Vdp.

    Se muestrea en cuantiles equiespaciados (no aleatoriamente) para que el
    resultado sea determinista y reproducible: el mismo Vdp siempre produce
    el mismo conjunto de capas.
    """
    if not (0.0 <= Vdp < 1.0):
        raise ValueError(f"Vdp debe estar en [0, 1); se recibió {Vdp}")
    if Vdp == 0.0:
        return np.full(n_layers, float(k_median))

    sigma = -np.log(1.0 - Vdp)
    p = (np.arange(n_layers) + 0.5) / n_layers      # cuantiles centrados
    z = norm.ppf(p)
    k = k_median * np.exp(sigma * z)
    return np.sort(k)[::-1]                          # de mayor a menor


def _front_position(k_ratio, M):
    """
    Posición adimensional del frente (u = x/L) en una capa de permeabilidad
    k_j, en el instante en que la capa de referencia k_i acaba de irrumpir.

    Integrando la ecuación de avance del frente para dos capas se obtiene:

        M*u + (1-M)*u^2/2 = (k_j/k_i) * (1+M)/2

    que es una cuadrática en u. Para M = 1 se reduce a u = k_j/k_i.
    """
    k_ratio = np.asarray(k_ratio, dtype=float)
    rhs = k_ratio * (1.0 + M) / 2.0

    if np.isclose(M, 1.0):
        u = rhs                     # M*u = rhs con M = 1
    else:
        a = (1.0 - M) / 2.0
        b = M
        c = -rhs
        disc = b * b - 4.0 * a * c
        disc = np.clip(disc, 0.0, None)
        u = (-b + np.sqrt(disc)) / (2.0 * a)

    return np.clip(u, 0.0, 1.0)


def vertical_sweep_dykstra_parsons(Vdp, M, wor_limit=25.0, n_layers=50):
    """
    Eficiencia de barrido vertical al alcanzar un WOR límite de operación.

    Procedimiento: se recorren las capas en orden de irrupción (de mayor a
    menor permeabilidad). En cada etapa se calcula la posición del frente en
    las capas aún no irrumpidas, la cobertura vertical acumulada y el WOR
    producido. Se devuelve la cobertura correspondiente al momento en que el
    WOR alcanza el límite económico.

    Parameters
    ----------
    Vdp : float
        Coeficiente de Dykstra-Parsons, [0, 1). 0 = homogéneo.
    M : float
        Relación de movilidad de punto extremo.
    wor_limit : float
        WOR económico límite (bbl agua / bbl petróleo). 25 corresponde
        aproximadamente a 96 % de corte de agua.
    n_layers : int
        Número de capas del modelo estratificado.

    Returns
    -------
    EV : float
        Eficiencia de barrido vertical en [0, 1].
    """
    if M <= 0:
        raise ValueError("M debe ser positiva.")

    k = _layer_permeabilities(Vdp, n_layers)

    EV_stages = []
    WOR_stages = []

    for i in range(n_layers):
        k_ref = k[i]
        u = _front_position(k / k_ref, M)
        u[: i + 1] = 1.0                      # capas ya irrumpidas

        EV_stages.append(float(np.mean(u)))

        # Caudales relativos: q_j ~ k_j / (u_j + (1 - u_j) * M)
        denom = u + (1.0 - u) * M
        q = k / denom
        q_water = float(np.sum(q[: i + 1]))    # capas irrumpidas -> agua
        q_oil = float(np.sum(q[i + 1:]))       # capas sin irrumpir -> petróleo

        WOR_stages.append(np.inf if q_oil <= 0 else q_water / q_oil)

    EV_stages = np.asarray(EV_stages)
    WOR_stages = np.asarray(WOR_stages)

    # Primera etapa que alcanza o supera el WOR límite
    idx = np.searchsorted(WOR_stages, wor_limit, side="left")

    if idx == 0:
        return float(EV_stages[0])
    if idx >= n_layers:
        return float(EV_stages[-1])

    # Interpolación lineal en WOR entre las dos etapas que lo encierran
    w0, w1 = WOR_stages[idx - 1], WOR_stages[idx]
    e0, e1 = EV_stages[idx - 1], EV_stages[idx]
    if not np.isfinite(w1) or np.isclose(w1, w0):
        return float(e1)

    frac = (wor_limit - w0) / (w1 - w0)
    return float(np.clip(e0 + frac * (e1 - e0), 0.0, 1.0))


# ======================================================================
#  RECOBRO TOTAL
# ======================================================================

def areal_sweep_after_breakthrough(M, qi_ratio):
    """
    Eficiencia de barrido areal DESPUÉS de la irrupción, patrón de cinco
    puntos (Craig et al., 1955):

        EA = EA_BT + 0.2749 * ln(Winj / Winj_BT)

    donde Winj/Winj_BT es la razón entre el agua acumulada inyectada y la
    inyectada hasta la irrupción. El coeficiente 0.2749 corresponde al
    patrón de cinco puntos.

    El barrido areal NO se detiene en la irrupción: el agua sigue ampliando
    el área contactada mientras se inyecta. Evaluar EA a la irrupción cuando
    ED y EV se evalúan al límite económico mezclaría estados distintos del
    mismo proceso y subestimaría el recobro.

    Parameters
    ----------
    M : float
        Relación de movilidad de punto extremo.
    qi_ratio : float
        Winj / Winj_BT (>= 1). Un valor de 1 devuelve el barrido a la
        irrupción.

    Returns
    -------
    EA : float, acotada a [EA_BT, 1.0]
    """
    EA_BT = float(areal_sweep_breakthrough(M))
    qi_ratio = max(float(qi_ratio), 1.0)
    EA = EA_BT + 0.2749 * np.log(qi_ratio)
    return float(np.clip(EA, EA_BT, 1.0))


def total_recovery_factor(ED, M, Vdp, qi_ratio=1.0,
                          wor_limit=25.0, n_layers=50):
    """
    Recobro total como producto de las tres eficiencias:

        RF_total = ED * EA * EV

    Las tres se evalúan en el MISMO instante del proceso: el momento en que
    el corte de agua producido alcanza el límite económico.

      ED : Buckley-Leverett / Welge al corte de agua límite.
      EA : Craig, corregido después de la irrupción con la razón de
           volúmenes inyectados qi_ratio = Qi_límite / Qi_irrupción.
      EV : Dykstra-Parsons al mismo WOR límite.

    Pasar qi_ratio = 1 devuelve el barrido areal a la irrupción, que es una
    cota inferior conservadora del recobro.

    Returns
    -------
    dict con RF_total, ED, EA, EV, EA_BT
    """
    EA_BT = float(areal_sweep_breakthrough(M))
    EA = areal_sweep_after_breakthrough(M, qi_ratio)
    EV = vertical_sweep_dykstra_parsons(Vdp, M, wor_limit, n_layers)
    RF = float(ED) * EA * EV

    return {
        "RF_total": float(np.clip(RF, 0.0, 1.0)),
        "ED": float(ED),
        "EA": EA,
        "EA_BT": EA_BT,
        "EV": EV,
    }
