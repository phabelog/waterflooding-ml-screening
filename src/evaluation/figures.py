"""
Figuras de la Etapa 5. Todas se guardan a 300 dpi y se distinguen también
por trama o estilo de línea, para que se lean en escala de grises.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

import train_models as tm         # noqa: E402
from evaluate import NOMBRES_VARIABLES   # noqa: E402

AZUL, NARANJA, GRIS, OSCURO = "#2B6A94", "#C8702A", "#8A8F98", "#14213D"
ESTILOS = {"arbol_decision": ("-.", GRIS), "random_forest": ("--", NARANJA),
           "xgboost": ("-", AZUL), "svm": (":", OSCURO)}
CORTO = {"arbol_decision": "Árbol de decisión", "random_forest": "Random Forest",
         "xgboost": "XGBoost", "svm": "SVM"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight"})


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_confusion(resultados, outdir):
    modelos = list(resultados["modelos"])
    fig, axes = plt.subplots(1, len(modelos), figsize=(2.6 * len(modelos), 2.9))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, modelos):
        c = resultados["modelos"][m]["matriz_confusion_prueba"]
        M = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.15)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]}", ha="center", va="center", fontsize=12,
                        color="white" if M[i, j] > M.max() * 0.55 else OSCURO, fontweight="bold")
        ax.set_xticks([0, 1], ["No apto", "Apto"])
        ax.set_yticks([0, 1], ["No apto", "Apto"] if ax is axes[0] else ["", ""])
        ax.set_xlabel("Predicción"); ax.grid(False)
        ax.set_title(CORTO[m] + (" (final)" if m == resultados["modelo_final"] else ""), fontsize=9.5)
    axes[0].set_ylabel("Etiqueta real")
    return _save(fig, outdir, "etapa5_matrices_confusion.png")


def fig_roc(resultados, extras, outdir):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for m, e in extras.items():
        ls, col = ESTILOS[m]
        auc = resultados["modelos"][m]["prueba"]["auc_roc"]
        ax.plot(e["roc"]["fpr"], e["roc"]["tpr"], ls=ls, color=col, lw=1.8,
                label=f"{CORTO[m]} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#BBBBBB", lw=0.8)
    ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos (recall)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005); ax.legend(loc="lower right", frameon=False)
    return _save(fig, outdir, "etapa5_curvas_roc.png")


def fig_train_val_test(resultados, outdir):
    modelos = list(resultados["modelos"])
    conj = [("entrenamiento", "Entrenamiento", "#C9D6E2", ""), ("validacion", "Validación cruzada", AZUL, ""),
            ("prueba", "Prueba", OSCURO, "//")]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=False)
    for ax, met, tit in zip(axes, ("f1", "auc_roc"), ("F1 de la clase apta", "AUC-ROC")):
        x = np.arange(len(modelos)); w = 0.26
        for k, (c, lab, col, hatch) in enumerate(conj):
            v = [resultados["modelos"][m][c][met] for m in modelos]
            b = ax.bar(x + (k - 1) * w, v, w, label=lab, color=col, edgecolor="white", hatch=hatch)
            for xi, vi in zip(x + (k - 1) * w, v):
                ax.text(xi, vi + 0.004, f"{vi:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
        lo = min(resultados["modelos"][m][c][met] for m in modelos for c, *_ in conj)
        ax.set_ylim(max(0, lo - 0.08), 1.045); ax.set_xticks(x, [CORTO[m].replace(" de ", " de\n").replace("Random ", "Random\n") for m in modelos], fontsize=8)
        ax.set_yticks([t for t in ax.get_yticks() if t <= 1.0001])
        ax.set_title(tit, fontsize=10); ax.grid(axis="x", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.06), fontsize=8.5)
    return _save(fig, outdir, "etapa5_entrenamiento_validacion_prueba.png")


def fig_importance(imp, modelo_final, outdir):
    d = imp.sort_values("importancia")
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    col = [NARANJA if c else AZUL for c in d["control_negativo"]]
    barras = ax.barh([NOMBRES_VARIABLES[v] for v in d.index], d["importancia"], xerr=d["desv"],
                     color=col, error_kw={"lw": 0.8, "ecolor": "#555555"})
    for barra, c in zip(barras, d["control_negativo"]):
        if c:
            barra.set_hatch("//"); barra.set_edgecolor("white")
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_xlabel(f"Caída del F1 al permutar la variable ({CORTO[modelo_final]}, conjunto de prueba)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AZUL, label="Variable del modelo físico"),
                       Patch(facecolor=NARANJA, hatch="//", edgecolor="white", label="Control negativo")],
              frameon=False, loc="lower right")
    return _save(fig, outdir, "etapa5_importancia_variables.png")


def fig_vs_traditional(resultados, outdir):
    fin = resultados["modelo_final"]
    series = [("Clasificador trivial", resultados["clasificador_trivial_prueba"], "#D5D8DC", ""),
              ("Cribado por rangos", resultados["cribado_tradicional_prueba"], NARANJA, "//"),
              (CORTO[fin] + " (modelo final)", resultados["modelos"][fin]["prueba"], AZUL, "")]
    mets = [("precision", "Precisión"), ("recall", "Recall"), ("f1", "F1"), ("accuracy", "Exactitud")]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    x = np.arange(len(mets)); w = 0.26
    for k, (lab, m, col, hatch) in enumerate(series):
        v = [m[a] for a, _ in mets]
        ax.bar(x + (k - 1) * w, v, w, label=lab, color=col, edgecolor="white", hatch=hatch)
        for xi, vi in zip(x + (k - 1) * w, v):
            ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [b for _, b in mets]); ax.set_ylim(0, 1.16); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=8)
    ax.set_ylabel("Valor en el conjunto de prueba")
    return _save(fig, outdir, "etapa5_ml_vs_cribado_tradicional.png")


def fig_reference(ref_df, resumen, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    ax = axes[0]
    apt = ref_df["label_fisica"] == 1
    ax.scatter(ref_df.loc[~apt, "M"], 100 * ref_df.loc[~apt, "RF_total"], s=6, color=NARANJA, marker="x",
               linewidths=0.6, label="No apto (criterio físico)")
    ax.scatter(ref_df.loc[apt, "M"], 100 * ref_df.loc[apt, "RF_total"], s=6, color=AZUL, label="Apto (criterio físico)")
    ax.axvline(10, color="#555555", lw=0.8, ls="--"); ax.axhline(15, color="#555555", lw=0.8, ls="--")
    ax.set_xlabel("Relación de movilidad M"); ax.set_ylabel("Recobro RF (%)"); ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=7.5, markerscale=2)
    ax = axes[1]
    ax.hist(ref_df["puntaje_modelo"], bins=25, color=AZUL, edgecolor="white")
    ax.axvline(0.5, color="#555555", lw=0.8, ls="--")
    ax.set_xlabel("Probabilidad de «apto» según el modelo"); ax.set_ylabel("Realizaciones")
    ax.set_title(f"Aptas: {100 * resumen['fraccion_apta_modelo']:.0f} % (modelo) · "
                 f"{100 * resumen['fraccion_apta_fisica']:.0f} % (criterio físico)", fontsize=8.5)
    return _save(fig, outdir, "etapa5_caso_referencia.png")


def make_all(resultados, objetos, outdir):
    paths = [fig_confusion(resultados, outdir),
             fig_roc(resultados, objetos["extras"], outdir),
             fig_train_val_test(resultados, outdir),
             fig_importance(objetos["importancia"], resultados["modelo_final"], outdir),
             fig_vs_traditional(resultados, outdir),
             fig_reference(objetos["referencia"], resultados["caso_referencia"], outdir)]
    return paths
