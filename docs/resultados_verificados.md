# Resultados verificados · tres clasificadores

Ejecución local del 17 de octubre de 2026. Regresión logística, árbol de decisión
y Naive Bayes para Churn, según confirmación de la autora. Reproducción:
`python -m src.report --output resultados/integrados`.

## Datos y preparación

Tres CSV inmutables: 2.395+2.394+2.514=7.303 filas, 21 columnas y 7.043 clientes.
Se conservan los hashes, procedencia y controles de calidad en el informe.
Se detectan 222 copias exactas incluidas en 260 filas adicionales por cliente;
38 clientes con diferencias en la entrada (36 tras normalizar), 1.534 ausentes,
152 variantes de contrato, 39 contratos desconocidos y 95 cargos negativos.
La consolidación por primera aparición deja 7.043 clientes; después se marcan
94 cargos negativos. Extremos e imputación se ajustan solo en entrenamiento,
incluido cada pliegue. La vista preparada tiene cero nulos, duplicados y errores
según los controles definidos, con 19 predictores y 46 columnas codificadas.
Los diagramas finales de atípicos son descriptivos y no exigen cero extremos.

## Validación y selección

5.634 clientes de entrenamiento; cinco pliegues estratificados y semilla 42.
Los tres algoritmos comparten la partición y los pliegues. Se selecciona por F1.

| Modelo | F1 medio (validación) | Desviación F1 | Recall medio (validación) | ROC-AUC medio (validación) |
| --- | --- | --- | --- | --- |
| Regresión logística | 0,6301 | 0,0155 | 0,7940 | 0,8423 |
| Árbol de decisión | 0,6138 | 0,0061 | 0,7786 | 0,8265 |
| Naive Bayes | 0,5955 | 0,0111 | 0,8455 | 0,8166 |

Modelo seleccionado: **Regresión logística**.

Parámetros del mejor candidato de cada familia:

```json
{
  "Regresión logística": {
    "modelo__C": 0.1,
    "modelo__class_weight": "balanced"
  },
  "Árbol de decisión": {
    "modelo__class_weight": "balanced",
    "modelo__max_depth": 8,
    "modelo__min_samples_leaf": 50
  },
  "Naive Bayes": {
    "modelo__var_smoothing": 0.001
  }
}
```

## Prueba final

1.409 clientes, incluidos 374 abandonos. Umbral fijo 0,50. No se utiliza prueba
para volver a elegir parámetros o familia.

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Regresión logística | 0,7445 | 0,5125 | 0,7647 | 0,6137 | 0,8274 |
| Árbol de decisión | 0,7353 | 0,5009 | 0,7406 | 0,5976 | 0,8058 |
| Naive Bayes | 0,6891 | 0,4525 | 0,8155 | 0,5821 | 0,8040 |
| Referencia mayoritaria | 0,7346 | 0,0000 | 0,0000 | 0,0000 | 0,5000 |

Matrices: filas reales, columnas predichas; orden [permanencia, abandono].

```text
Regresión logística: [[763, 272], [88, 286]]
Árbol de decisión: [[759, 276], [97, 277]]
Naive Bayes: [[666, 369], [69, 305]]
Referencia mayoritaria: [[1035, 0], [374, 0]]
```

Los gráficos de cada subpestaña muestran ROC, precisión/sensibilidad, matriz y
probabilidades por clase real. La comparación incluye la referencia mayoritaria.
La interpretación de coeficientes, importancias del árbol y medias condicionales
de Naive Bayes se mantiene separada; no se atribuye causalidad.

## Verificación

`python -m pytest -q`: **18 pruebas superadas y 2 omitidas**. Las omisiones
corresponden al módulo generador histórico retirado y a su comprobación opcional;
no se omiten pruebas del flujo activo. Se verifican integración, hashes, conflictos,
preparación, ausencia de fuga, métricas, los tres algoritmos y navegación de las
subpestañas con predicción individual. Informe CSV/JSON actualizado correctamente.

No se ha realizado inspección visual manual en navegador ni ejecución nativa de
Windows. Las probabilidades no están calibradas para uso operativo y una prueba
aleatoria no demuestra generalización temporal, causalidad o retorno de inversión.
