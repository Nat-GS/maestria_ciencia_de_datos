# Telco · Análisis y predicción del abandono

Proyecto final de Natalia Gutiérrez Soliz · Maestría en Ciencia de Datos · UCB.
Aplicación local en español para presentar las primeras cuatro etapas de CRISP-DM.
Integra los **tres archivos sucios entregados**, audita su calidad, muestra la
preparación y compara regresión logística, árbol de decisión y Naive Bayes.

## Ejecutar en Windows

Python 3.12 de 64 bits recomendado (entorno verificado: 3.12.3). Desde PowerShell
o CMD, en la carpeta del proyecto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abrir http://localhost:8501 y detener con Ctrl+C. No hace falta activar scripts
ni cambiar la política de PowerShell. Si `py` no está disponible, usar `python`
después de verificar la versión. Internet para instalar dependencias; después
el análisis funciona con archivos locales, sin Colab, Drive ni credenciales.
La ejecución nativa de Windows no se ha comprobado desde este entorno Linux.

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m streamlit run app.py
```

## Recorrido de la aplicación

- **Inicio:** título del proyecto, objetivo general, CSV de data/, lectura e integración, vista por archivo y diccionario de datos.
- **Exploración:** gráficos y análisis, filtros y controles de calidad de los archivos integrados.
- **Limpieza y preparación:** unión de CSV y limpieza con código Python, información de campos, cálculos y gráficos de duplicados/nulos/valores incorrectos; tabla limpia y controles finales calculados.
- **Modelado:** subpestañas Regresión logística, Árbol de decisión, Naive Bayes y Comparación; KPIs, matrices, ROC, precisión/sensibilidad y probabilidades por resultado real.
- **Predicción:** formulario de ocho variables y tres modelos específicos reentrenados y evaluados con ese subconjunto; probabilidad, bandas de riesgo y asociaciones descriptivas de entrenamiento.
- **Conclusiones:** resultados y limitaciones.

La navegación presenta los datos sin copiar literalmente cada apartado del informe.
Las versiones de Exploración distinguen datos recibidos, clientes sin repetir con
faltantes y datos limpios con faltantes completados. Este último caso significa
reemplazar ausentes con estadísticas aprendidas solo en entrenamiento, no recuperar
los valores reales originales.

Las interpretaciones se recalculan en Python, sin servicios externos. Los filtros
solo afectan la exploración. Los modelos se entrenan cuando una sección necesita
sus resultados y se conservan en memoria durante la sesión.

## Datos y decisiones principales

Entradas obligatorias, que no se sobrescriben:

| Archivo | Filas | Columnas |
|---|---:|---:|
| data/telco_csv_1.csv | 2.395 | 21 |
| data/telco_csv_2.csv | 2.394 | 21 |
| data/telco_csv_3.csv | 2.514 | 21 |

Integración vertical: **7.303 registros y 7.043 clientes únicos**. Hay 222 copias
exactas, 38 clientes con diferencias, 1.534 celdas ausentes, variantes de contrato,
95 cargos negativos y valores extremos positivos. Se normalizan variantes
inequívocas y se conserva la primera fila por cliente, con registro de conflictos.
Las etiquetas contradictorias bloquean el flujo. No se utiliza el original para
restaurar información. Las tres fuentes incluyen 260 filas extra por ID; después
de consolidar quedan 7.043 clientes y 19 predictores.

Separación estratificada 80/20 y semilla 42. Dentro de cada entrenamiento se aprende
el límite de extremos de cargo mensual Q3+1,5×RIC, después se imputan medianas/modas,
se codifican categorías y se escalan numéricas solo para logística. Selección por
F1 de cinco pliegues. Prueba final: exactitud, precisión, recall, F1, ROC-AUC y matriz,
con umbral 0,50. No se seleccionan parámetros usando prueba.
La vista preparada descargable sirve para auditoría; no debe reutilizarse como
entrada para entrenar saltándose los pipelines.

## Pruebas e informe sin interfaz

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.report --output resultados/integrados
```

El informe contiene inventario con hashes, procedencia, calidad, acciones,
conflictos, estadísticas, frecuencias, KPIs, vista preparada, búsqueda y métricas.
El JSON registra parámetros, versiones, límites, medianas e IDs de las particiones.
`resultados/` se excluye de Git porque se regenera. Las dependencias directas están
en `requirements.txt`; `requirements-lock.txt` fija el entorno completo verificado.

## Organización

```text
app.py                            Interfaz y navegación
src/business.py                   Contexto y KPIs
src/integration.py                Tres fuentes, procedencia y consolidación
src/data_processing.py            Tipos, calidad y preprocesamiento aprendido
src/exploration.py                KPIs y diccionario
src/interpretation.py              Gráficos y explicaciones automáticas
src/modeling.py                    Validación, entrenamiento y predicción
src/report.py                      Informe reproducible sin interfaz
data/telco_csv_1.csv             Primera entrada
data/telco_csv_2.csv             Segunda entrada
data/telco_csv_3.csv             Tercera entrada
docs/metodologia.md                Decisiones por etapa y subapartado
docs/revision_academica.md         Conciliación histórica y flujo actual
docs/resultados_verificados.md     Resultados y verificaciones actuales
docs/referencias/                  Notebooks y evidencia de referencia
tests/                            Datos, integración, modelos e interfaz
AGENTS.md                          Contexto para continuar el trabajo
```

Referencia de exploración: [notebook de la autora](https://colab.research.google.com/drive/1XvwNXWW5_AGOKTid3WT7Dfc16hh7h-vz).
Fuente de origen: [Telco Customer Churn, BlastChar / Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
El notebook analizaba una base anterior de 7.183 filas; se reproducen sus métodos,
recalculados sobre los tres archivos actuales. No se deben confundir sus cifras.
`src/generate_dirty_data.py` y el notebook de ensuciamiento se conservan como
referencia histórica; no son necesarios para usar la aplicación. Los CSV antiguos
ya no están en la carpeta actual y no se recrean automáticamente.

Las conclusiones son académicas, sin inferencia causal, calibración operativa,
validación temporal ni retorno de inversión demostrado. El DOCX no se modifica
sin revisión de la autora. No se realizan commits ni push automáticamente.
