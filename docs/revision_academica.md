# Estado vigente: integración de tres CSV

La aplicación actual usa exclusivamente `telco_csv_1.csv`, `telco_csv_2.csv`
y `telco_csv_3.csv`. Suman 7.303 registros, no los 7.183 del notebook histórico.
No se conoce el código que creó estas nuevas alteraciones; se auditan directamente
los archivos entregados, sin atribuirles semillas o porcentajes no verificados.

| Evidencia actual | Resultado |
|---|---:|
| Filas por archivo | 2.395 / 2.394 / 2.514 |
| Columnas por archivo | 21 |
| Clientes únicos | 7.043 |
| Copias exactas integradas | 222 |
| Filas adicionales por ID | 260 |
| Clientes con campos diferentes antes/después de normalizar | 38 / 36 |
| Ausentes de antigüedad / cargo mensual / cargo total / contrato | 555 / 487 / 373 / 119 |
| Cargos mensuales negativos | 95 |
| Variantes de contrato normalizables | 152 |
| Contrato «Desconocido» | 39 |

La normalización y consolidación dejan 7.043 clientes. Se conserva la primera fila
por ID y se registran conflictos; no se afirma que sea la versión verdadera.
Tras consolidar se convierten 94 cargos negativos a ausentes. Los extremos positivos
se tratan con límites del entrenamiento y las imputaciones se ajustan dentro de
cada pliegue. No se recuperan valores desde el original.

El notebook de exploración `1XvwNXWW5_AGOKTid3WT7Dfc16hh7h-vz` analizaba el CSV
histórico de 7.183 filas. Sus análisis se reutilizan, con cifras recalculadas sobre
las nuevas entradas. Deben actualizarse tablas y texto académico para distinguir
fuentes integradas, clientes consolidados y datos imputados. Las nuevas métricas
se documentan en resultados_verificados.md; el orden de las fuentes cambia la
partición concreta respecto a experimentos previos, aunque se conserve la semilla.
No comparar esas ejecuciones como si compartieran la misma prueba.

Los archivos históricos de datos ya no están en la carpeta actual; no se recrean.
La evidencia que sigue corresponde a la verificación efectuada previamente contra
Drive, conservada para explicar el origen de las cifras del DOCX original.
El DOCX no se modifica automáticamente.

---

# Revisión académica y conciliación de registros

Se conserva sin editar `TF_GUTIERREZ_SOLIZ_NATALIA_2026.docx` y su copia
`docs/trabajo_final.docx`. La primera revisión solo disponía del original y no
podía explicar la diferencia. El notebook aportado posteriormente permite cerrarla.

## Fuente del procedimiento

[Ensuciando_datos.ipynb](https://colab.research.google.com/drive/1HK9Qsc4s_ua4o7uI-J6FBDtY4bpNCykE),
copia íntegra en `referencias/Ensuciando_datos.ipynb`.
La celda de carga usa el [CSV original de Drive](https://drive.google.com/file/d/1Rkq42BS8xWVIvIFzFW83wlWA3nrlRk6f/view).
La salida guardada del notebook enlaza el [CSV sucio de Drive](https://drive.google.com/file/d/1o1Ilpxk69avNJ0rP6DAooWObG6fE-EZ1/view).
Ambos se descargaron solo para verificar su equivalencia con los archivos locales;
no se modificaron archivos de Drive. Los enlaces son referencias, no dependencias
de ejecución. Evidencia con hashes en `referencias/verificacion_colab.json`.

## Explicación exacta de las 140 filas

El código crea `np.random.default_rng(42)`, convierte TotalCharges a número e
introduce nulos en el 5 % de los valores disponibles de tenure, MonthlyCharges y
TotalCharges, en ese orden. Usa `int` y muestreo sin reemplazo. Después selecciona
140 filas mediante `int(7043 * 0.02)` con el mismo RNG y las concatena al final.
Son copias de filas ya alteradas, incluidos sus nulos; no son clientes nuevos.

| Evidencia | Original | Sucio local | DOCX |
|---|---:|---:|---:|
| Registros | 7.043 | 7.183 | 7.183 |
| Clientes únicos | 7.043 | 7.043 | No indicado |
| Filas duplicadas | 0 | 140 | No indicado |
| Abandonaron | 1.869 | 1.913 | 1.913 |
| Permanecieron | 5.174 | 5.270 | 5.270 |
| Género masculino | 3.555 | 3.627 | 3.627 |
| Género femenino | 3.488 | 3.556 | 3.556 |
| Contrato mensual | 3.875 | 3.956 | 3.956 |
| Contrato de un año | 1.473 | 1.502 | 1.502 |
| Contrato de dos años | 1.695 | 1.725 | 1.725 |
| tenure disponible | 7.043 | 6.824 | 6.824 |
| MonthlyCharges disponible | 7.043 | 6.824 | 6.824 |
| TotalCharges disponible | 7.032 | 6.815 | 6.815 |

Las copias añaden 44 abandonos y 96 permanencias. La tasa sucia es 26,63 %;
la original, 26,54 %. Los conteos de las tablas numéricas del documento corresponden
al sucio, no a una inconsistencia adicional. Se reproducen también sus medias
redondeadas: tenure 32,27; MonthlyCharges 64,61; TotalCharges 2.283,36.

## Nulos y transformaciones verificadas

| Variable | Vacíos previos reconocidos | Nulos nuevos | Nulos copiados | Total sucio |
|---|---:|---:|---:|---:|
| tenure | 0 | 352 | 7 | 359 |
| MonthlyCharges | 0 | 352 | 7 | 359 |
| TotalCharges | 11 | 351 | 6 | 368 |

No se introducen inconsistencias de servicios, errores categóricos o valores
atípicos. Se convierte TotalCharges de texto a número y tenure pasa a float por
los nulos. No se alteran valores observados ni etiquetas. Los 11 vacíos iniciales
son campos con espacios; se reconocen como ausentes al normalizar, aunque
`isna()` sobre la lectura original sin conversión no los cuente.

**Verificación binaria realizada:** original local idéntico al original de Drive;
sucio local idéntico al CSV guardado por Colab. No solo coinciden dimensiones y
resúmenes: coinciden bytes, orden, valores y posiciones nulas.

SHA-256 original:
`88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`.
SHA-256 sucio:
`9513c385194c67fc51ab70d5af7f33df578179b75a43f4b511e38b62897cacfe`.

## Correcciones propuestas al documento

Conservar las cifras de 7.183 para describir explícitamente el **dataset sucio
con copias** y sus resultados exploratorios previos a limpieza. Añadir el
procedimiento reproducible, distinguir registros de clientes únicos y documentar
la selección de 7.043 clientes tras quitar 140 copias antes de separar 80/20.
Describir la imputación de 352, 352 y 362 ausentes tras deduplicar, ajustada
exclusivamente en entrenamiento. La modelación usa este flujo, sin restaurar los
valores ocultados desde el original.

Se mantienen el objetivo general, los cuatro objetivos específicos y los seis
KPIs. Por acuerdo de la autora, sustituir referencias de ejecución en Colab por
Python local y la regresión lineal de MonthlyCharges por dos clasificadores de
Churn; MonthlyCharges queda como predictor. Colab se cita como origen del
procedimiento de ensuciamiento, no como requisito de ejecución. Streamlit presenta
las etapas 1–4 sin afirmar implantación empresarial. La portada, bibliografía y
textos de plantilla aún requieren revisión. No se edita el DOCX automáticamente.
