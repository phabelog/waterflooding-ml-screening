"""
Interfaz gráfica (Etapa 6): cribado técnico de reservorios candidatos a
inyección de agua (waterflooding).

Ejecutar en local:   streamlit run app/streamlit_app.py

El modelo se entrena al iniciar la aplicación con el conjunto de
entrenamiento y los hiperparámetros de la Etapa 4, de modo que no depende
de archivos binarios ni de versiones de librerías.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ, "src", "evaluation"))

import evaluate as ev                                     # noqa: E402
import preprocess as pp                                   # noqa: E402
import train_models as tm                                 # noqa: E402
from generate_dataset import FEATURE_COLUMNS, compute_physics   # noqa: E402
from labeling import THRESHOLDS, evaluate_scenario        # noqa: E402
from variable_ranges import (INPUT_VARIABLES, RELPERM_VARIABLES,   # noqa: E402
                             water_viscosity_from_temperature)

st.set_page_config(page_title="Cribado de candidatos a waterflooding", page_icon="💧", layout="wide")


@st.cache_resource(show_spinner="Entrenando el modelo (solo la primera vez)…")
def cargar_modelo():
    df = pd.read_csv(os.path.join(RAIZ, "data", "synthetic", "synthetic_dataset.csv"))
    e4 = tm.load_results(os.path.join(RAIZ, "results", "etapa4_resultados.json"))
    disponibles = {k: v for k, v in e4["modelos"].items()
                   if k != "xgboost" or tm.xgboost_available()}
    nombre = ev.select_final_model({"modelos": disponibles})
    fitted, split = ev.fit_final_models(df, e4, models=[nombre])
    X_train = split[0]
    return fitted[nombre], nombre, X_train.min(), X_train.max()


modelo, nombre_modelo, x_min, x_max = cargar_modelo()

st.title("Cribado técnico de candidatos a inyección de agua")
st.caption("Modelo de machine learning entrenado con 4 000 reservorios sintéticos generados con un modelo "
           "físico (Corey, Buckley-Leverett, Darcy, Craig y Dykstra-Parsons). Maestría en Petróleos, ESPOL.")

with st.expander("Cómo usar esta herramienta y qué significa el resultado"):
    st.markdown(
        "- Ingrese las propiedades del reservorio y pulse **Evaluar**.\n"
        "- El resultado indica si el reservorio es un **candidato apto** para un estudio detallado de "
        "inyección de agua, con la probabilidad que le asigna el modelo.\n"
        "- Es un **primer filtro técnico**: no reemplaza la simulación numérica ni la evaluación económica.\n"
        "- El modelo es válido dentro de los rangos con que se entrenó; si un valor queda fuera, se le avisará.\n"
        "- El recobro se expresa respecto del petróleo presente al inicio de la inyección.")


def rango(var):
    lo, hi = (INPUT_VARIABLES.get(var) or RELPERM_VARIABLES.get(var))[:2]
    return float(lo), float(hi)


c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Roca")
    k = st.number_input("Permeabilidad k (mD)", 0.1, 20000.0, 500.0, step=10.0, help="Entrenamiento: 5 – 3 000 mD")
    phi = st.number_input("Porosidad φ (fracción)", 0.01, 0.45, 0.16, step=0.01, format="%.2f", help="Entrenamiento: 0.08 – 0.25")
    h = st.number_input("Espesor neto h (ft)", 1.0, 500.0, 40.0, step=1.0, help="Entrenamiento: 5 – 80 ft")
    vdp = st.slider("Coeficiente de Dykstra-Parsons Vdp", 0.0, 1.0, 0.60, 0.01, help="Entrenamiento: 0.30 – 0.90. Mayor valor = más heterogeneidad")
    depth = st.number_input("Profundidad vertical (ft)", 1000.0, 20000.0, 9500.0, step=50.0, help="Entrenamiento: 7 000 – 10 500 ft")
    skin = st.number_input("Skin del pozo inyector", -6.0, 50.0, 0.0, step=0.5, help="Entrenamiento: −2 a 8")
with c2:
    st.subheader("Fluidos y saturaciones")
    api = st.number_input("Gravedad API (°API)", 5.0, 60.0, 24.0, step=0.5, help="Entrenamiento: 14 – 35 °API")
    muo = st.number_input("Viscosidad del petróleo en reservorio μo (cP)", 0.1, 1000.0, 3.5, step=0.1, help="Entrenamiento: 0.3 – 70 cP")
    modo_muw = st.radio("Viscosidad del agua", ["Estimar con la temperatura", "Ingresar el valor"], horizontal=True)
    if modo_muw == "Estimar con la temperatura":
        T = st.number_input("Temperatura del reservorio (°F)", 60.0, 350.0, 205.0, step=1.0, help="Entrenamiento: 180 – 230 °F")
        muw = float(water_viscosity_from_temperature(T))
        st.caption(f"Viscosidad del agua estimada: {muw:.3f} cP")
    else:
        muw = st.number_input("Viscosidad del agua μw (cP)", 0.1, 2.0, 0.31, step=0.01, format="%.3f", help="Entrenamiento: 0.26 – 0.38 cP")
    swi = st.slider("Saturación de agua irreducible Swi", 0.0, 0.60, 0.25, 0.01, help="Entrenamiento: 0.10 – 0.45")
    sor = st.slider("Saturación residual de petróleo Sor", 0.0, 0.60, 0.25, 0.01, help="Entrenamiento: 0.15 – 0.35")
    so = st.slider("Saturación actual de petróleo So", 0.0, 1.0, 0.65, 0.01, help="Debe estar entre Sor y 1 − Swi")
with c3:
    st.subheader("Permeabilidad relativa (Corey)")
    kro = st.slider("kro máximo (en Swi)", 0.0, 1.0, 0.80, 0.01, help="Entrenamiento: 0.60 – 0.95")
    krw = st.slider("krw máximo (en 1 − Sor)", 0.0, 1.0, 0.30, 0.01, help="Entrenamiento: 0.15 – 0.55")
    no = st.slider("Exponente de Corey del petróleo no", 1.0, 6.0, 2.5, 0.1, help="Entrenamiento: 1.5 – 4.0")
    nw = st.slider("Exponente de Corey del agua nw", 1.0, 6.0, 2.5, 0.1, help="Entrenamiento: 1.5 – 4.0")

caso = {"k_md": k, "phi": phi, "h_net_ft": h, "api": api, "Swi": swi, "Sor": sor, "Vdp": vdp,
        "depth_ft": depth, "skin": skin, "kro_max": kro, "krw_max": krw, "no": no, "nw": nw,
        "muo_cp": muo, "muw_cp": muw, "So_current": so}

errores = []
if 1.0 - swi - sor < 0.10:
    errores.append("La saturación móvil (1 − Swi − Sor) debe ser de al menos 0.10.")
if not (sor <= so <= 1.0 - swi + 1e-9):
    errores.append(f"La saturación actual de petróleo debe estar entre Sor ({sor:.2f}) y 1 − Swi ({1 - swi:.2f}).")

if st.button("Evaluar", type="primary"):
    if errores:
        for e in errores:
            st.error(e)
        st.stop()

    X = pd.DataFrame([caso])[FEATURE_COLUMNS]
    pred = int(modelo.predict(X)[0])
    prob = float(modelo.predict_proba(X)[0, 1]) if hasattr(modelo, "predict_proba") else None

    fuera = [c for c in FEATURE_COLUMNS if X[c].iloc[0] < x_min[c] or X[c].iloc[0] > x_max[c]]

    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        if pred == 1:
            st.success("### Candidato APTO")
        else:
            st.error("### Candidato NO APTO")
        if prob is not None:
            st.metric("Probabilidad de «apto» según el modelo", f"{100 * prob:.0f} %")
            if 0.35 <= prob <= 0.65:
                st.info("El caso está cerca de la frontera de decisión: conviene revisarlo con más detalle.")
        st.caption(f"Modelo: {tm.MODEL_LABELS[nombre_modelo]}.")
    with r2:
        if fuera:
            st.warning("**Valores fuera del rango de entrenamiento** (el modelo extrapola; lea el resultado con cautela): "
                       + ", ".join(ev.NOMBRES_VARIABLES[c] for c in fuera) + ".")
        else:
            st.caption("Todas las propiedades están dentro del rango de entrenamiento.")

        try:
            fis = compute_physics(pd.Series(caso))
            eti = evaluate_scenario(caso, fis)
            th = THRESHOLDS
            filas = [
                ("Permeabilidad k", f"{k:,.0f} mD", f"≥ {th['k_min_md']:.0f} mD", k >= th["k_min_md"]),
                ("Espesor neto h", f"{h:.0f} ft", f"≥ {th['h_net_min_ft']:.0f} ft", h >= th["h_net_min_ft"]),
                ("Saturación actual So", f"{so:.2f}", f"≥ {th['So_min']:.2f}", so >= th["So_min"]),
                ("Viscosidad del petróleo μo", f"{muo:.1f} cP", f"≤ {th['muo_max_cp']:.0f} cP", muo <= th["muo_max_cp"]),
                ("Relación de movilidad M", f"{fis['M']:.1f}", f"≤ {th['M_max']:.0f}", fis["M"] <= th["M_max"]),
                ("Coeficiente Vdp", f"{vdp:.2f}", f"≤ {th['Vdp_max']:.2f}", vdp <= th["Vdp_max"]),
                ("Recobro RF", f"{100 * fis['RF_total']:.1f} %", f"≥ {100 * th['RF_total_min']:.0f} %", fis["RF_total"] >= th["RF_total_min"]),
                ("Caudal de inyección", f"{fis['q_inj_bpd']:,.0f} bbl/d", f"≥ {th['injectivity_min_bpd']:.0f} bbl/d", fis["q_inj_bpd"] >= th["injectivity_min_bpd"]),
            ]
            st.markdown("**Diagnóstico con el modelo físico** (cálculo directo, para explicar el resultado)")
            st.dataframe(pd.DataFrame([{"Criterio": a, "Valor": b, "Umbral": c, "Cumple": "Sí" if d else "No"}
                                       for a, b, c, d in filas]), hide_index=True)
            st.caption(f"Eficiencias: desplazamiento ED = {fis['ED']:.2f}, areal EA = {fis['EA']:.2f}, "
                       f"vertical EV = {fis['EV']:.2f}. Criterio físico: {'apto' if eti['label'] else 'no apto'}.")
            if eti["label"] != pred:
                st.info("El modelo y el criterio físico no coinciden en este caso: está cerca de un umbral.")
        except Exception as exc:   # el cálculo físico puede fallar con combinaciones extremas
            st.warning(f"No se pudo calcular el diagnóstico físico para esta combinación ({exc}).")

st.divider()
st.caption("Herramienta de cribado técnico con fines académicos. Código y metodología: "
           "github.com/phabelog/waterflooding-ml-screening")
