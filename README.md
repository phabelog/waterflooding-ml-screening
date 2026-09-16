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
de desplazamiento de Buckley-Leverett), para el cribado técnico (screening)
de reservorios candidatos a proyectos de inyección de agua (waterflooding).

## Metodología

El proyecto sigue seis etapas (ver `docs/`):

1. **Generación y recolección de datos** — framework físico, definición de
   variables de entrada, generación de escenarios sintéticos, etiquetado
   apto/no apto, recolección de datos reales para validación externa.
2. **Análisis de datos** — estadística descriptiva, correlación, balance de
   clases, consistencia física.
3. **Preprocesamiento** — limpieza, división estratificada, escalado, SMOTE
   (solo en entrenamiento).
4. **Desarrollo de modelos** — árbol de decisión (base), Random Forest,
   XGBoost, SVM, optimización de hiperparámetros.
5. **Evaluación y validación** — métricas de clasificación, matriz de
   confusión, interpretabilidad, comparación con métodos tradicionales de
   screening, validación externa con datos reales de campo.
6. **Interfaz gráfica** — aplicación en Streamlit para ingreso de parámetros
   y predicción (Apto/No apto) con probabilidad asociada.

## Estructura del repositorio

```
├── src/
│   ├── physics/            # Framework físico: Corey, Buckley-Leverett, Darcy
│   ├── dataset_generation/ # Generación del dataset sintético etiquetado
│   ├── preprocessing/      # Limpieza, split, escalado, SMOTE
│   ├── models/             # Entrenamiento y evaluación de modelos ML
│   └── app/                # Aplicación Streamlit (Etapa 6)
├── notebooks/               # Notebooks de exploración (Google Colab)
├── data/
│   ├── synthetic/           # Dataset sintético generado
│   └── real_field_validation/  # Datos reales de EP Petroecuador (anonimizados,
│                                # publicados con autorización expresa de la
│                                # institución)
├── docs/                    # Figuras, diagramas de metodología
└── tests/                   # Verificación del framework físico
```

## Framework físico — estado actual

El módulo `src/physics/` implementa y verifica:

- **Corey**: permeabilidades relativas kro(Sw), krw(Sw).
- **Buckley-Leverett / Welge**: curva de flujo fraccional, localización del
  frente de choque (dos métodos numéricos independientes de verificación
  cruzada), saturación promedio a la irrupción, curva de recobro vs.
  volúmenes porosos inyectados.
- **Darcy**: relación de movilidad end-point, índice de inyectividad radial.

Verificación: `python src/physics/validate.py` (chequea condiciones de
frontera, consistencia cruzada entre métodos, balance de materia y
sensibilidad física; genera `docs/validation_plot.png`).

## Cómo ejecutar

```bash
pip install -r requirements.txt
python src/physics/validate.py
```

Recomendado: desarrollo en Google Colab (notebooks en `notebooks/`,
sincronizados con este repositorio) para evitar requerimientos de cómputo
local. Ver `notebooks/` para los cuadernos de generación de dataset y
entrenamiento de modelos conforme se desarrollen.

## Datos

Los datos reales utilizados como conjunto de validación externa (*blind
validation*) fueron proporcionados por EP Petroecuador (Gerencia de Activo
Auca) en el marco de un convenio de cooperación con ESPOL, y se publican en
este repositorio de forma anonimizada con autorización expresa de la
institución.

## Citación

*(Pendiente — se completará al momento de la publicación del artículo, con
DOI de Zenodo del release correspondiente al código y datos usados en el
manuscrito.)*

## Licencia

Ver [LICENSE](LICENSE).
