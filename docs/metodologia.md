# Metodología académica · flujo de tres archivos

Natalia Gutiérrez Soliz · Maestría en Ciencia de Datos · UCB · MCD-701.
Se implementan las primeras cuatro etapas de CRISP-DM. La evaluación técnica está
incluida en modelado. Predicción y conclusiones son demostración y síntesis del
trabajo; no constituyen implantación empresarial de las etapas 5 y 6.
El DOCX se conserva sin editar. El flujo histórico de ensuciamiento se documenta
en `revision_academica.md`; la entrada actual son los tres CSV entregados.

## 1. Comprensión del negocio

### 1.1 Objetivos
Analizar factores asociados al abandono y estimar riesgo a partir de características
personales, servicios, contrato, antigüedad y cargos para orientar retención.
Se mantienen los cuatro objetivos del DOCX: revisar calidad, explorar asociaciones,
estimar probabilidades y evaluar la identificación de clientes en riesgo.
La unidad de modelado es el cliente; Churn=Yes es la clase positiva.
Un falso negativo omite un abandono; un falso positivo puede generar contacto
innecesario. Se priorizan recall y F1, informando también precisión.
No hay meta comercial numérica validada, costos ni resultados de campañas para
calcular retorno de inversión o afirmar reducción real de abandono.

### 1.2 Recursos, requisitos, supuestos y restricciones
Python local, Pandas, NumPy, Scikit-learn, Plotly, Streamlit, tres archivos CSV,
notebooks de referencia y documento académico. Internet solo para instalación
y consulta de referencias; el dashboard y el informe no acceden a Drive.
Se supone disponibilidad de predictores al estimar riesgo. No hay fechas,
geografía, historial de tarifas ni información de campañas. Las asociaciones
no son efectos causales. La resolución de conflictos no identifica una versión
histórica verdadera; es una política reproducible con auditoría.

### 1.3 Plan y KPIs
Secuencia: negocio → lectura e integración → EDA y calidad → preparación →
validación de modelos → presentación. Cronograma orientativo de siete sesiones:
negocio (1), datos (2–3), preparación (4), modelado (5–6), presentación (7).
Se verifican los módulos antes de completar la interfaz.
KPIs del DOCX: abandono general, por contrato, por intervalo de antigüedad, por
internet y por intervalo de cargo mensual; promedio de cargos por clase de abandono.
Cada tasa divide los abandonos por las filas del grupo y el promedio solo usa
cargos disponibles. En la vista sucia, las repeticiones cuentan como registros,
no como nuevos clientes. Los filtros no modifican la muestra de modelado.
Intervalos: antigüedad 0–12, 13–24, 25–48, 49–72; cargos hasta 35, más de 35 a 70,
más de 70 a 100 y más de 100. Los ausentes aparecen como «Sin dato».

## 2. Comprensión de los datos

### 2.1 Recolección e integración de lectura
Fuente de referencia: [Telco Customer Churn, BlastChar, Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Entrada efectiva: `data/telco_csv_1.csv` (2.395 filas), `telco_csv_2.csv`
(2.394) y `telco_csv_3.csv` (2.514). Se conservan intactos; no se generan datos
aleatorios nuevos. Los hashes constan en el inventario exportado y las pruebas.
Se concatenan verticalmente en orden de nombre 1→2→3, con esquema exacto de 21
columnas. No se hace join por ID. Archivo y fila CSV se guardan en una tabla
paralela, fuera de los predictores. Se obtienen 7.303 filas y 7.043 clientes.

### 2.2 Descripción
Un identificador, 19 predictores y Churn. Numéricas: tenure, MonthlyCharges y
TotalCharges. SeniorCitizen es categórica binaria. El resto describe características
personales, servicios, contratos y pagos. No se atribuye moneda o periodo calendario
que no figure en los datos. El dashboard incluye diccionario y vista de cada fuente.

### 2.3 Exploración y correspondencia con Colab
Referencia: [Exploracion.ipynb](https://colab.research.google.com/drive/1XvwNXWW5_AGOKTid3WT7Dfc16hh7h-vz).
Instantánea local en `docs/referencias/Exploracion.ipynb`; no se ejecutan lecturas
remotas o código de autenticación. El notebook analizaba el CSV sucio histórico
único de 7.183 filas. Se reproducen sus análisis sobre las nuevas 7.303 filas,
por lo que los valores actuales no deben coincidir con sus salidas históricas.

| Celda (índice desde cero) | Análisis | Implementación local |
|---|---|---|
| 2 | Describe de tres numéricas y mediana | numeric_statistics |
| 3 | Frecuencias de todas las categóricas, incluidos ausentes | frequency_table |
| 4 | Tres histogramas de 25 intervalos | numeric_exhibit, conteos con numpy.histogram |
| 5 | Conteos de género, adulto mayor, contrato, internet, pago y Churn | category_exhibit, seis barras horizontales |
| 6 | Crosstab contrato × Churn, totales y tasa | contract_exhibit |
| 7 | Diccionario de variables | data_dictionary |

Plotly sustituye Matplotlib/Seaborn. En los conteos gráficos de categorías se
omiten ausentes, como countplot; en sus tablas se incluyen con denominador total,
como value_counts(dropna=False). La tabla cruzada de contratos omite ausentes,
como pd.crosstab del notebook, y comunica cuántos se excluyen.
Las vistas integradas mantienen variantes; las consolidadas normalizan dominios.
Cada gráfico tiene interpretación determinista calculada sobre los mismos datos
y filtros: qué representa, conteos, extremos, medias/medianas o tasas, y relación
con abandono. No son textos de un servicio externo ni conclusiones prefijadas.
Los histogramas comparan abandono hasta la mediana y por encima, con tamaños de
grupo. Las categóricas muestran frecuencia principal y tasa más alta, sin atribuir
causalidad. También se muestran ausentes, comparación visual antes/después,
métricas, matriz de confusión y curvas ROC con explicaciones numéricas.

### 2.4 Calidad observada
Hay 1.534 celdas ausentes: tenure 555, MonthlyCharges 487, TotalCharges 373,
Contract 119. Se detectan 222 copias exactas y 260 filas adicionales por ID;
38 clientes tienen campos diferentes antes de normalizar (36 después).
Contrato: 152 variantes inequívocas, 39 «Desconocido» y 119 vacíos.
MonthlyCharges: 95 negativos y 125 valores superiores a Q3+1,5×RIC de la muestra
integrada. Este último conteo es auditoría descriptiva y no determina el límite
del modelo. No hay inconsistencias entre telefonía/internet y servicios dependientes.
No se deduce el procedimiento generador de estos errores: se describen los
valores realmente presentes en los CSV, sin asumir porcentajes o semillas nuevas.

## 3. Preparación

### 3.1 Selección
Conservar 19 predictores y una fila por cliente. Excluir customerID del aprendizaje;
Churn: No=0, Yes=1. Archivo y fila no entran en el modelo. Los originales no se
usan para recuperar valores ocultos o elegir la fila «correcta».

### 3.2 Limpieza
Normalizar espacios y tipos. Contrato se pasa a minúsculas solo para buscar el
mapa explícito: month-to-month / month to month → Month-to-month;
one year → One year; two year / two years → Two year. Otras categorías como
Desconocido pasan a ausentes; no se supone qué contrato tenía el cliente.
Consolidar por customerID reteniendo la primera ocurrencia según archivo y fila.
La política es independiente del objetivo y de las métricas. Se registra cada
conflicto y posición conservada. Etiquetas Churn contradictorias bloquean el
entrenamiento, en lugar de resolverse silenciosamente. Al quitar 260 filas quedan
7.043 clientes. Los 222 duplicados exactos ya están incluidos en ese conteo.
Después de consolidar hay 94 cargos negativos; pasan a NaN. No se borran clientes
por nulos. Quedan 538 ausentes de tenure, 569 de MonthlyCharges, 362 de TotalCharges
y 157 de Contract antes del tratamiento aprendido de extremos.

### 3.3 Estructuración y transformaciones aprendidas
Partición estratificada 80/20, semilla 42, sobre los clientes consolidados:
5.634 entrenamiento, 1.409 prueba. Cada pipeline aprende, dentro de cada pliegue,
el límite Q3+1,5×RIC de MonthlyCharges usando únicamente su entrenamiento.
Los cargos superiores pasan a ausentes; después se imputan numéricas por mediana,
categóricas por moda y se codifican mediante one-hot. StandardScaler solo para
las numéricas de logística. El árbol no requiere escalado. El tratamiento de
extremos es una elección metodológica revisable, sin afirmar que todo extremo
positivo sea un error confirmado. No se aplica al objetivo ni se filtran filas.
Tras seleccionar parámetros se reajustan transformaciones con todo entrenamiento,
nunca con prueba. La vista preparada del dashboard muestra imputación sin codificar
y distingue particiones; no se reutiliza para entrenar fuera del pipeline.
Una mediana de antigüedad puede ser fraccionaria y no constituye observación real.

### 3.4 Integración final
ColumnTransformer combina numéricas y categóricas, manteniendo correspondencia
entre filas y etiquetas. La procedencia queda auditada fuera de la matriz. Las
agrupaciones descriptivas de antigüedad/cargo no se agregan como predictores.

## 4. Modelado

### 4.1 Algoritmos
Regresión logística binaria regularizada L2 y árbol de decisión de complejidad
controlada. Referencia adicional: DummyClassifier de clase mayoritaria.

### 4.2 Plan de prueba
Misma partición y cinco pliegues estratificados (shuffle=True, semilla 42).
Selección de parámetros por F1 medio de validación y familia por F1, con desempate
por recall y luego ROC-AUC; se registran recall,
ROC-AUC y desviación de F1. Umbral fijo 0,50. Ninguna búsqueda se ajusta con prueba.
No se usa SMOTE; class_weight se compara dentro de validación.

### 4.3 Construcción
GridSearchCV, refit por F1, n_jobs=1. Logística: C en {0.1,1,10}, class_weight en
{None,balanced}, max_iter=2000. Árbol: max_depth en {3,5,8}, min_samples_leaf en
{20,50}, class_weight en {None,balanced}. Se exportan búsqueda, parámetros,
versiones, hashes de entradas, índices e IDs de ambas particiones.

### 4.4 Evaluación técnica
Exactitud, precisión, recall, F1, ROC-AUC, curvas ROC y matriz con filas reales,
columnas predichas y orden [permanencia,abandono]. Precision y F1 sin alertas
positivas se definen como cero. El baseline muestra el efecto del desbalance.
Coeficientes logísticos y reducción de impureza del árbol son medidas distintas,
no efectos causales. Una prueba aleatoria no establece generalización temporal;
las probabilidades no tienen calibración operativa. Las conclusiones se calculan
con el modelo seleccionado por validación y sus verdaderos/falsos positivos y
negativos, sin cambiar la selección a partir del test.

## Reproducción
`python -m pytest -q` prueba fuentes, integración, procedencia, conflictos,
preparación, límites aprendidos solo en entrenamiento, métricas, interpretaciones,
navegación y predicción. `python -m src.report --output resultados/integrados`
genera inventario, calidad, acciones, conflictos, frecuencias, estadísticas,
KPIs, vista preparada, búsqueda, métricas y metadatos sin Streamlit.
El generador histórico permanece como referencia opcional y no se ejecuta al
abrir la aplicación. Las pruebas actuales no requieren los CSV históricos retirados.

## Presentación simplificada de la aplicación
La interfaz reúne título, objetivo general, archivos, lectura e integración y
diccionario en Inicio. Exploración incluye los controles y gráficos de calidad.
Se conservan Limpieza y preparación, Modelado, Predicción y Conclusiones sin
numeración de subapartados del informe. Las versiones de datos explican en lenguaje
común si los faltantes permanecen o fueron completados con estadísticas aprendidas
en entrenamiento. Los cálculos, modelos y fuentes permanecen iguales.

## Demostración de limpieza paso a paso
Limpieza y preparación muestra primero código de concatenación de los tres CSV,
filas recibidas y controles de cada campo. Después compara duplicados antes y
después, nulos por campo (incluida la etapa de marcar valores incorrectos), y
errores de escritura/dominio o rangos. El orden visual nulos/errores no cambia la
dependencia real: los errores se marcan ausentes antes de completar los faltantes.
Los fragmentos explicativos se acompañan del código de las funciones utilizadas.
`src/cleaning.py` construye vistas usando las mismas transformaciones del modelo,
sin volver a ajustar estadísticas. La tabla final tiene 7.043 filas y 21 columnas,
sin la columna auxiliar de partición. Se verifican nulos y valores incorrectos
por campo, y duplicados de fila/cliente e inconsistencias de servicios en un
control general. Repetir categorías entre distintos clientes no es un duplicado
inválido. Un negativo puede ser válido para integer/float, pero incumple el dominio
de cargos o antigüedad. Los ceros finales se calculan, no se fijan manualmente.
El informe exporta también controles_finales.csv y controles_finales_por_campo.csv.

## Correspondencia visual con el documento académico
Las tablas y figuras actualizadas de recolección, exploración y calidad se
incorporan como capturas reales de Streamlit, con los datos recibidos y sin filtros.
Las capturas mantienen encabezados, categorías, redondeo, colores y ejes de la web.
Las frecuencias se presentan por variable, igual que en el selector del dashboard.
El manifiesto de docs/figuras/capturas_aplicacion/ registra fuentes, estado y hashes.
Se verifica la identidad binaria de cada imagen incrustada con su captura.

Al final de Limpieza y preparación se comparan diagramas de caja de antigüedad,
cargo mensual y cargo total antes/después sobre los mismos clientes. Los puntos
y conteos usan límites descriptivos Q1−1,5×RIC y Q3+1,5×RIC de cada versión.
No se reutilizan para ajustar modelos ni se exige cero atípicos: un extremo
puede ser válido. El tratamiento aprendido en entrenamiento permanece intacto.

## Modelado con tres clasificadores
La autora confirmó regresión logística para Churn en lugar de regresión lineal.
Se añade GaussianNB con var_smoothing en {1e-9,1e-6,1e-3}, seleccionado por F1
en los mismos cinco pliegues. Usa la misma partición de clientes y preprocesamiento
sin fuga: tratamiento de extremos, imputación, one-hot y escalado numérico.
GaussianNB aproxima cada entrada transformada con una distribución gaussiana e
independencia condicional; en indicadores one-hot es una aproximación, no un
modelo específico para variables categóricas. No se presupone calibración.
La selección de familia incluye los tres modelos con el mismo criterio de F1.
El diagrama de la subpestaña Árbol de decisión representa el estimador completo
ya ajustado, con profundidad visible configurable sin reentrenar ni podar.
`src/tree_diagram.py` usa sus reglas, Gini y clases ponderadas; calcula los
conteos reales por clase recorriendo exclusivamente clientes de entrenamiento.
Los conteos observados no se confunden con las proporciones/pesos de tree.value.
Los nodos truncados se identifican; las reglas one-hot se muestran traducidas.
El botón «Descargar árbol en PNG» rasteriza en el navegador el SVG real de ese
componente, conservando profundidad, etiquetas y colores. No recrea el gráfico
con otra biblioteca, no requiere servicios externos y no modifica el modelo.
El dashboard separa los algoritmos en subpestañas y reúne su comparación, con
texto mínimo y gráficos ROC, precisión/sensibilidad, matrices, distribución de
probabilidades y diagnósticos de predictores. Las medias condicionales de Naive
Bayes se muestran como tales y no se presentan como importancias del árbol.
La predicción individual permite elegir cualquiera de los tres clasificadores.

La caché de entrenamientos incluye una firma del código de modelos, preparación
e integración. Si cambia la implementación, se reentrena. Antes de mostrar las
subpestañas se comprueba la presencia de los tres modelos en estimadores, métricas,
validación, parámetros y predicciones; una entrada incompleta se descarta y
reentrena automáticamente, evitando reutilizar experimentos antiguos de dos modelos.

En Resumen numérico, «Estadísticas descriptivas de las variables numéricas»
muestra los ocho estadísticos como filas y tenure, MonthlyCharges, TotalCharges
como columnas. Pandas calcula count, mean, std muestral, min, cuartiles y max,
excluyendo nulos por variable. El formato español utiliza dos decimales salvo
count, que se muestra entero; el formato visual no modifica los datos numéricos.
La interpretación y la descarga se recalculan con la misma versión y filtros
del gráfico; por defecto corresponden a los tres CSV integrados sin filtros.

La sección «Comparación de datos antes y después de la limpieza» utiliza
`src/cleaning_comparison.py` sobre las etapas y el pipeline ya ajustados, sin
aprender parámetros adicionales. Compara registros, copias exactas (incluidas
en las filas adicionales por ID), faltantes iniciales/finales y su evolución,
histogramas con intervalos comunes y cajas de MonthlyCharges. Identifica los
negativos y extremos tratados con el límite aprendido en entrenamiento. El
gráfico de contratos conserva todas las filas antes/después del mapeo vigente
para aislar la normalización de la consolidación y del completado posterior.
El resumen sí compara la entrada con la vista final de clientes únicos y
distingue sus columnas originales de los predictores codificados. Los valores
inválidos se cuentan con field_checks, incluyendo extremos mensuales según el
límite entrenado; faltantes se informan por separado. Todas las cifras se
recalculan desde las fuentes actuales; estas vistas no se usan para entrenar.

## Predicción simplificada y evidencia de modelado

`src/prediction.py` calcula importancias por permutación compatibles con los tres
algoritmos: clona cada pipeline seleccionado, lo reajusta con el 80 % interno del
entrenamiento y calcula caída de F1 en el 20 % restante, con cinco permutaciones
y semilla 42. No consulta la prueba final. Las configuraciones ya fueron elegidas
con el entrenamiento completo: esta reserva se usa para explicación y selección,
no como una validación independiente de esas configuraciones. La permutación
puede infravalorar predictores correlacionados y generar combinaciones de servicios
incoherentes; no establece causalidad.

Se seleccionan ocho entradas: antigüedad y cargo mensual como base numérica para
conservar el filtro mensual validado, y otras seis por caída media de F1 entre
los tres modelos. Actualmente son SeniorCitizen, InternetService, TechSupport,
Contract, PaymentMethod y TotalCharges. La lista se recalcula desde entrenamiento,
no se fija en el formulario. No se seleccionan variables con métricas de prueba.
La selección previa no está anidada en los cinco pliegues posteriores; las métricas
de validación reducidas pueden ser optimistas. Una mejora futura es anidar la
selección en validación cruzada y comprobar estabilidad de las variables.

`run_experiment(features=...)` entrena tres pipelines específicos de ocho entradas,
con las mismas particiones, pliegues, rejillas y transformaciones del experimento
completo. Conserva el orden numérico tenure, MonthlyCharges y, si se selecciona,
TotalCharges para reutilizar el filtro original sin modificarlo. Cada pliegue
aprende límite, medianas, modas, codificación y escalado solamente con entrenamiento.
El modelo completo de 19 entradas y sus resultados se presentan por separado.

Las bandas ilustrativas son bajo p<0,30, medio 0,30≤p<0,60 y alto p≥0,60; no son
umbrales comerciales optimizados ni probabilidades calibradas. La clasificación
binaria de evaluación conserva 0,50. Los valores predeterminados del formulario
son medianas y modas de entrenamiento. Los servicios se validan para mantener
coherencia con internet y teléfono cuando ambas variables están seleccionadas.

La explicación individual compara tasas observadas de grupos del entrenamiento:
misma categoría o tramo respecto a la mediana numérica válida. No son atribuciones
locales del modelo. Los cargos superiores al límite entrenado se completan por
el pipeline y no se interpretan como cargos observados válidos. Las recomendaciones
de retención son orientaciones generales sin efectividad ni ahorros demostrados.

La evidencia al final de Modelado distingue demostración local implementada de
propuestas futuras: integración con CRM, campañas controladas, validación temporal,
calibración, monitoreo y reentrenamiento. No se ejecutan etapas empresariales 5–6
de CRISP-DM ni se modifican capítulos o índice de la monografía.
