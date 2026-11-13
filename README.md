# Modelo de Machine Learning para el cribado técnico de reservorios candidatos a waterflooding

Repositorio de código del artículo científico desarrollado como proyecto de
titulación de la Maestría en Petróleos (ESPOL), en cooperación con EP
Petroecuador (Gerencia de Activo Auca).

**Alcance.** El modelo se entrena y se evalúa únicamente con datos
sintéticos generados por el framework físico. Los datos de campo facilitados
por EP Petroecuador se usan solo como referencia (contexto), de forma
anonimizada; este trabajo no incluye una validación de campo.

**Autores:** Phabell Andre Diaz Klinger, Juan Joseph Martinez Alvarez
**Tutor:** Ing. Freddy Paul Carrión Maldonado, Ph.D.
**Institución:** Escuela Superior Politécnica del Litoral (ESPOL)

## Objetivo

Desarrollar un modelo de Machine Learning, informado por un framework físico
de flujo bifásico (ley de Darcy, permeabilidades relativas de Corey, teoría
de desplazamiento de Buckley-Leverett), para el cribado técnico (*screening*)
de reservorios candidatos a proyectos de inyección de agua (*waterflooding*).

## Metodología y estado de avance

| Etapa | Contenido | Estado |
|---|---|---|
| 1. Generación de datos | Framework físico, variables de entrada, 5 000 escenarios sintéticos, etiquetado apto/no apto en dos niveles | Completada |
| 2. Análisis de datos | Estadística descriptiva, correlación, balance de clases, consistencia física | Completada |
| 3. Preprocesamiento | Atípicos y transformación logarítmica, división estratificada, escalado dentro del pipeline, ponderación de clases (sin SMOTE) | Completada |
| 4. Desarrollo de modelos | Árbol de decisión (base), Random Forest, XGBoost, SVM, optimización de hiperparámetros | Completada |
| 5. Evaluación | Evaluación única en el conjunto de prueba, métricas (F1, precisión, recall, exactitud, AUC-ROC), importancia de variables y controles negativos, comparación con el cribado por rangos, caso de referencia | Completada |
| 6. Interfaz gráfica | Aplicación en Streamlit: ingreso de las 16 propiedades, predicción, probabilidad y diagnóstico físico | Completada |

## Estructura del repositorio

```
├── src/
│   ├── physics/            # Framework físico: Corey, Buckley-Leverett, Darcy, eficiencias de barrido
│   ├── dataset_generation/ # Variables, rangos, etiquetado y generación del dataset sintético
│   ├── analysis/           # Análisis y auditoría del dataset (Etapa 2)
│   ├── preprocessing/      # Limpieza, división, escalado, tratamiento del desbalance (Etapa 3)
│   ├── models/             # Construcción y optimización de los cuatro modelos (Etapa 4)
│   └── evaluation/         # Evaluación, importancia de variables, cribado por rangos, caso de referencia y figuras (Etapa 5)
├── app/                    # Aplicación Streamlit (Etapa 6)
├── notebooks/              # Cuadernos ejecutables (Google Colab)
│   ├── 01_framework_fisico.ipynb
│   ├── 02_analisis_datos.ipynb
│   ├── 03_preprocesamiento.ipynb
│   ├── 04_desarrollo_modelos.ipynb
│   └── 05_evaluacion_validacion.ipynb
├── data/
│   └── synthetic/          # Dataset sintético generado (5 000 escenarios etiquetados)
├── results/                # Hiperparámetros y validación cruzada (Etapa 4); resultados de la evaluación (Etapa 5)
├── docs/                   # Figuras y documentación de la metodología
└── tests/                  # Verificaciones automatizadas
```

## Framework físico

El módulo `src/physics/` implementa:

- **Corey** — permeabilidades relativas `kro(Sw)` y `krw(Sw)`.
- **Buckley-Leverett / Welge** — curva de flujo fraccional, localización del
  frente de choque (dos métodos numéricos independientes para verificación
  cruzada), saturación promedio a la irrupción, curva de recobro frente a
  volúmenes porosos inyectados y eficiencia de desplazamiento al corte de
  agua económico.
- **Darcy** — relación de movilidad de punto extremo e índice de
  inyectividad radial.
- **Eficiencias de barrido** — barrido areal mediante la correlación de
  Willhite (1986) sobre la curva de Craig (1971) para patrón de cinco
  puntos, con la corrección posterior a la irrupción de Craig et al. (1955),
  y barrido vertical resolviendo directamente el modelo
  estratificado de Dykstra y Parsons (1950).

El recobro total se calcula como el producto de las tres eficiencias
(`RF = ED · EA · EV`), lo que sitúa los resultados en el rango realista de un
proyecto de inyección de agua en lugar de la eficiencia de desplazamiento
unidimensional aislada.

## Dataset sintético

`data/synthetic/synthetic_dataset.csv` contiene 5 000 escenarios etiquetados
(38.9 % aptos / 61.1 % no aptos).

**Separación entre variables de entrada y etiqueta.** El modelo se entrena
únicamente con las variables crudas del reservorio. Las características
derivadas del framework físico (`ED`, `M`, `EA`, `EV`, `RF_total`,
`q_inj_bpd`) se emplean exclusivamente para construir la etiqueta y se
conservan en el archivo solo con fines de trazabilidad: entregarlas como
entradas haría que el modelo reprodujera la regla de etiquetado en lugar de
aprender la relación entre las propiedades del reservorio y su aptitud.

## Verificación

```bash
pip install -r requirements.txt
python tests/test_physics_framework.py    # framework físico
python tests/test_dataset_generation.py   # eficiencias de barrido, etiquetado y dataset
python tests/test_data_analysis.py        # análisis y auditoría del dataset (Etapa 2)
python tests/test_preprocessing.py        # preprocesamiento sin fuga de información (Etapa 3)
python tests/test_models.py               # modelos, grillas y validación cruzada (Etapa 4)
python tests/test_evaluation.py           # evaluación, cribado por rangos y caso de referencia (Etapa 5)
```

Todos los conjuntos de verificaciones deben superarse en su totalidad.

## Cómo ejecutar

Los cuadernos de `notebooks/` pueden ejecutarse en línea (por ejemplo, en
Google Colab) o en un entorno local. Cada cuaderno descarga el repositorio y reproduce en línea las figuras
y resultados de su etapa.

Para regenerar el dataset sintético:

```bash
python src/dataset_generation/generate_dataset.py
```

## Resultados (Etapa 5)

Modelo final: **XGBoost**, seleccionado por su F1 medio en validación
cruzada. Evaluación única sobre el conjunto de prueba (1 000 escenarios que
ningún modelo vio durante el ajuste):

| Modelo | F1 entrenamiento | F1 validación cruzada | F1 prueba | Exactitud prueba | AUC-ROC prueba |
|---|---|---|---|---|---|
| XGBoost (final) | 1.000 | 0.953 ± 0.008 | 0.944 | 0.956 | 0.993 |
| Random Forest | 1.000 | 0.932 ± 0.011 | 0.919 | 0.936 | 0.984 |
| Árbol de decisión (base) | 0.940 | 0.912 ± 0.014 | 0.895 | 0.917 | 0.961 |
| SVM (RBF) | 0.909 | 0.839 ± 0.008 | 0.843 | 0.874 | 0.950 |
| Cribado tradicional por rangos | — | — | 0.784 | 0.786 | — |
| Clasificador trivial (todo apto) | — | — | 0.560 | 0.389 | 0.500 |

- **Importancia por permutación:** viscosidad del petróleo, saturación actual
  de petróleo, coeficiente de Dykstra-Parsons y permeabilidad. Los controles
  negativos (porosidad y profundidad) obtienen importancia prácticamente nula.
- **Frente al cribado por rangos:** los falsos positivos bajan de 214 a 29.
- Todos los valores están en `results/etapa5_resultados.json` y las figuras
  en `docs/figures/`.

Las métricas miden la capacidad del clasificador para reproducir el criterio
del framework físico; no son una validación frente a resultados de campo.

## Aplicación web (Etapa 6)

**Aplicación en línea:** https://waterflooding-ml-screening.streamlit.app

Para ejecutarla en un equipo local:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

La aplicación entrena el modelo final al iniciar, a partir del dataset y de
los hiperparámetros publicados (no usa archivos binarios). El usuario ingresa
las 16 propiedades del reservorio y obtiene la clasificación, la probabilidad,
una advertencia si algún valor queda fuera del dominio de entrenamiento y el
diagnóstico físico (relación de movilidad, recobro y caudal de inyección
frente a sus umbrales).

## Datos

El entrenamiento y la evaluación usan **únicamente el dataset sintético**.

Los datos de campo facilitados por EP Petroecuador, bajo acuerdo de
confidencialidad, corresponden a un campo maduro del Activo Auca y se emplean
solo como **referencia**:

1. **Contraste de rangos.** Los valores medidos (permeabilidad, porosidad,
   espesor, gravedad API, profundidad, saturaciones extremas) se comparan con
   los rangos del dataset sintético.
2. **Caso de referencia.** Con las propiedades medidas fijas y las no medidas
   sorteadas en su intervalo, se generan 2 000 realizaciones que se clasifican
   con el criterio físico y con el modelo (coincidencia: 91.8 %). Es un
   ejercicio de contexto, **no una validación de campo**.

Los valores de referencia se publican solo de forma agregada y anonimizada
(`src/evaluation/evaluate.py`). Los archivos originales **no se publican**:
`.gitignore` bloquea por defecto `data/real_field_validation/`.

## Citación y uso

Este repositorio acompaña a un artículo científico en desarrollo. El método,
el framework físico y el criterio de cribado aquí implementados son obra de
los autores indicados.

Si utiliza este software, el conjunto de datos o el método que implementan
—total o parcialmente, o en una reimplementación propia— por favor cite el
trabajo. GitHub genera la referencia automáticamente a partir del archivo
`CITATION.cff` mediante el botón **"Cite this repository"**, en la columna
derecha de la página principal.

**Referencia del artículo:** *(pendiente — se completará al momento de la
publicación).*

**Referencia del código y los datos:** *(pendiente — se asignará un DOI de
Zenodo al release correspondiente a la versión empleada en el manuscrito).*

El código se distribuye bajo licencia MIT (ver [LICENSE](LICENSE)), que
permite su uso, modificación y redistribución conservando el aviso de
autoría. El uso académico del método descrito se rige, además, por las normas
habituales de atribución y citación.

## Referencias del framework

- Buckley, S. E., & Leverett, M. C. (1942). *Mechanism of fluid displacement in sands*. Transactions of the AIME, 146(1), 107–116.
- Corey, A. T. (1954). *The interrelation between gas and oil relative permeabilities*. Producers Monthly, 19(1), 38–41.
- Craig, F. F. (1971). *The Reservoir Engineering Aspects of Waterflooding*. SPE Monograph Series, Vol. 3.
- Dykstra, H., & Parsons, R. L. (1950). *The prediction of oil recovery by water flood*. Secondary Recovery of Oil in the United States, 2, 160–174.
- Welge, H. J. (1952). *A simplified method for computing oil recovery by gas or water drive*. JPT, 4(4), 91–98.
- Willhite, G. P. (1986). *Waterflooding*. SPE Textbook Series, Vol. 3.

## Licencia

Ver [LICENSE](LICENSE).
