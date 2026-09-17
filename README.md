# Modelo de Machine Learning para el cribado técnico de reservorios candidatos a waterflooding

Repositorio de código del artículo científico desarrollado como proyecto de
titulación de la Maestría en Petróleos (ESPOL), en cooperación con EP
Petroecuador (Gerencia de Activo Auca).

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
| 1. Generación y recolección de datos | Framework físico, variables de entrada, escenarios sintéticos, etiquetado apto/no apto | Completada (salvo datos reales) |
| 2. Análisis de datos | Estadística descriptiva, correlación, balance de clases, consistencia física | Completada |
| 3. Preprocesamiento | Limpieza, división estratificada, escalado, SMOTE (solo en entrenamiento) | Pendiente |
| 4. Desarrollo de modelos | Árbol de decisión (base), Random Forest, XGBoost, SVM, optimización de hiperparámetros | Pendiente |
| 5. Evaluación y validación | Métricas, matriz de confusión, interpretabilidad, comparación con métodos tradicionales, validación externa | Pendiente |
| 6. Interfaz gráfica | Aplicación en Streamlit: ingreso de parámetros y predicción con probabilidad asociada | Pendiente |

## Estructura del repositorio

```
├── src/
│   ├── physics/            # Framework físico: Corey, Buckley-Leverett, Darcy, eficiencias de barrido
│   ├── dataset_generation/ # Variables, rangos, etiquetado y generación del dataset sintético
│   ├── analysis/           # Análisis y auditoría del dataset (Etapa 2)
│   ├── preprocessing/      # Limpieza, división, escalado, SMOTE (Etapa 3)
│   ├── models/             # Entrenamiento y evaluación de modelos (Etapa 4)
│   └── app/                # Aplicación Streamlit (Etapa 6)
├── notebooks/              # Cuadernos ejecutables (Google Colab)
│   ├── 01_framework_fisico.ipynb
│   └── 02_analisis_datos.ipynb
├── data/
│   ├── synthetic/          # Dataset sintético generado (5 000 escenarios etiquetados)
│   └── real_field_validation/  # Datos reales para validación externa (ver sección "Datos")
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
  puntos, y barrido vertical resolviendo directamente el modelo
  estratificado de Dykstra y Parsons (1950).

El recobro total se calcula como el producto de las tres eficiencias
(`RF = ED · EA · EV`), lo que sitúa los resultados en el rango realista de un
proyecto de inyección de agua en lugar de la eficiencia de desplazamiento
unidimensional aislada.

## Dataset sintético

`data/synthetic/synthetic_dataset.csv` contiene 5 000 escenarios etiquetados
(44.6 % aptos / 55.4 % no aptos).

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
```

Ambos conjuntos de verificaciones deben superarse en su totalidad.

## Cómo ejecutar

Desarrollo recomendado en Google Colab, con los cuadernos de `notebooks/`
sincronizados con este repositorio, para evitar requerimientos de cómputo
local. Cada cuaderno descarga el repositorio y reproduce en línea las figuras
y resultados de su etapa.

Para regenerar el dataset sintético:

```bash
python src/dataset_generation/generate_dataset.py
```

## Datos

El conjunto de validación externa (*blind validation*) provendrá de datos
reales de campo solicitados formalmente a EP Petroecuador (Gerencia de Activo
Auca), en el marco del convenio de cooperación con ESPOL. **La solicitud se
encuentra en trámite**; los datos aún no han sido recibidos y la carpeta
`data/real_field_validation/` está vacía.

Una vez recibidos, se publicarán en este repositorio de forma anonimizada,
sujeto a la autorización expresa de la institución. Hasta entonces,
`.gitignore` bloquea por defecto los archivos de esa carpeta para evitar
publicaciones involuntarias.

Los resultados de las Etapas 2 a 4 se obtienen sobre el dataset sintético y
no dependen de estos datos; la validación externa corresponde a la Etapa 5.

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
