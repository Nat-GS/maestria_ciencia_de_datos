# Contexto del proyecto

Trabajo final de Natalia Gutiérrez Soliz, Maestría en Ciencia de Datos, Universidad
Católica Boliviana «San Pablo», módulo Introducción a Ciencia de Datos (MCD-701).
Objetivo: analizar factores asociados al abandono y estimar el riesgo individual
para orientar estrategias de retención. No afirmar causalidad ni ahorros reales.

## Alcance acordado

Implementar las primeras cuatro etapas de CRISP-DM y todos sus subapartados.
La evaluación técnica pertenece al modelado; no se implementan las etapas 5 y 6.
Python local, Streamlit, Pandas, NumPy, Scikit-learn y Plotly. Interfaz y documentación
en español. Regresión logística, árbol de decisión y Naive Bayes, misma partición estratificada
80/20, semilla 42. Priorizar recall y F1; informar también precision, accuracy,
ROC-AUC y matriz de confusión. No utilizar Google Colab.

## Fuentes y decisiones vigentes

- Entrada activa: `data/telco_csv_1.csv`, `telco_csv_2.csv`, `telco_csv_3.csv`.
  No modificarlos ni generar alteraciones adicionales. Integrar verticalmente en
  orden 1→2→3: 2.395+2.394+2.514=7.303 filas, 21 columnas y 7.043 clientes únicos.
- `src/integration.py` conserva procedencia fuera de predictores, audita conflictos,
  normaliza variantes de Contract y conserva primera fila por ID según archivo/fila.
  260 filas extra incluyen 222 copias exactas; 38 clientes difieren en la entrada,
  36 tras normalizar. No elegir versiones usando la etiqueta o el original.
  Las etiquetas contradictorias bloquean el entrenamiento.
- Cargos negativos e inválidos pasan a ausentes; dentro de cada pipeline se aprende
  el límite superior Q3+1,5×RIC de MonthlyCharges solo con entrenamiento, antes de
  imputar. El límite descriptivo de auditoría no se utiliza para entrenar.
- Modelado: subpestañas Regresión logística, Árbol de decisión, Naive Bayes y
  Comparación. Priorizar gráficos y métricas con texto mínimo. La autora confirmó
  regresión logística para Churn, no regresión lineal de una variable numérica.
  Naive Bayes usa GaussianNB, con var_smoothing seleccionado dentro de validación.
- Modelos actuales: `run_experiment(raw, integrated=True)`. Misma partición de
  clientes y mismos cinco pliegues; selección por F1 y prioridad de recall.
- Notebook de exploración: `docs/referencias/Exploracion.ipynb`, ID de Drive
  `1XvwNXWW5_AGOKTid3WT7Dfc16hh7h-vz`. Reproducir sus tablas, tres histogramas
  de 25 intervalos, seis barras categóricas y crosstab contrato/Churn en Plotly,
  recalculados sobre las fuentes actuales. El notebook analizaba 7.183 filas antiguas.
- `src/interpretation.py` calcula interpretaciones, gráficos y conclusiones con datos
  reales, sin textos numéricos prefijados ni servicios externos.
- Navegación simplificada: Inicio, Exploración, Limpieza y preparación, Modelado,
  Predicción, Conclusiones. No reproducir literalmente los puntos del informe.
  Inicio contiene título, objetivo general, lectura e integración de CSV y diccionario.
  Calidad de datos está dentro de Exploración. Usar «valores faltantes completados»
  y explicar que son estimaciones de entrenamiento; evitar «imputados» sin explicación.
  El detalle CRISP-DM permanece en documentación, sin ampliar el alcance.
- Documento académico: `docs/trabajo_final.docx`; conservar sin editar.
- Historia: `generate_dirty_data.py` reproduce el ensuciamiento previo de Colab
  (semilla 42, nulos 5 %, copias 2 %). No es parte de la entrada actual. Los CSV
  históricos ya no están en data/ y no deben recrearse para abrir la aplicación.
  La evidencia previa explica 7.043+140=7.183 y se conserva en revision_academica.md.
- Limpieza y preparación debe mostrar unión con pd.concat, datos iniciales,
  información por campo, código Python, cálculos y gráficos antes/después para
  duplicados, nulos y valores incorrectos, más tabla final y controles calculados.
  No fijar ceros ni contar categorías válidas repetidas como errores.
- Vista preparada: solo auditoría; no reutilizarla para entrenar fuera de pipelines.
- Predicción simplificada: ocho entradas elegidas con entrenamiento, conservando
  tenure y MonthlyCharges y completando por importancia media de permutación
  entre los tres algoritmos. `src/prediction.py` entrena modelos específicos con
  el subconjunto y los mismos clientes, pliegues y rejillas. Nunca completar las
  entradas omitidas arbitrariamente para usar el modelo completo. La selección
  previa no está anidada en los pliegues: documentar posible optimismo de CV.
  Bandas ilustrativas: bajo <30 %, medio 30–<60 %, alto ≥60 %, sin calibración
  comercial. Selección de familia por F1 de validación; desempate recall y ROC-AUC.
- Informe actual: `python -m src.report --output resultados/integrados`.

## Convenciones

- Lógica independiente de Streamlit en `src/`; interfaz en `app.py`.
- Ajustar imputación, codificación y escalado solamente con entrenamiento y dentro
  de cada pliegue de validación. Nunca seleccionar parámetros con la prueba final.
- Conservar categorías «sin servicio» como información válida; excluir customerID.
- La lógica de lectura/modelado no depende de Streamlit; las interpretaciones
  automáticas deben usar los mismos datos y filtros que cada gráfico.
- No editar el DOCX original automáticamente. Documentar correcciones propuestas.
- Registrar decisiones y limitaciones en `docs/metodologia.md`.
- Las tablas y los gráficos añadidos al documento académico deben coincidir
  visualmente con los de Streamlit. Usar capturas reales de los mismos componentes,
  registrar versión de datos y filtros, y no recrearlos con otra biblioteca o
  agrupar tablas que la aplicación muestra por separado. Para la exploración
  inicial usar «Datos tal como se recibieron», sin filtros.
- Ejecutar `python -m pytest -q` y la prueba de Streamlit antes de dar por terminado.
- No hacer commits ni push sin aprobación explícita. No versionar `.venv/`.
