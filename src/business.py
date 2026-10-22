"""Contexto académico y plan; sin dependencia de la interfaz."""

OBJECTIVE = (
    "Analizar los factores asociados al abandono de clientes de telecomunicaciones "
    "y estimar su riesgo según características, servicios, antigüedad y cargos, "
    "para orientar estrategias de retención."
)
SPECIFIC_OBJECTIVES = [
    "Identificar y tratar problemas de calidad que afecten el análisis.",
    "Analizar asociaciones entre abandono, servicios, antigüedad, contrato y cargos.",
    "Estimar la probabilidad de abandono a partir de características del cliente.",
    "Evaluar la capacidad de identificar clientes con riesgo de abandono.",
]
PLAN = [
    ("1. Comprensión del negocio", "Objetivos, recursos, supuestos, restricciones, KPIs y plan"),
    ("2. Comprensión de los datos", "Recolección, descripción, exploración y calidad"),
    ("3. Preparación de los datos", "Selección, limpieza, estructuración e integración"),
    ("4. Modelado", "Algoritmos, plan de pruebas, construcción y evaluación técnica"),
]
KPI_DEFINITIONS = [
    ("Abandono general", "100 × abandonos / clientes"),
    ("Abandono por contrato", "100 × abandonos del contrato / clientes del contrato"),
    ("Abandono por antigüedad", "100 × abandonos del intervalo / clientes del intervalo"),
    ("Abandono por internet", "100 × abandonos del servicio / clientes del servicio"),
    ("Abandono por cargo mensual", "100 × abandonos del intervalo / clientes del intervalo"),
    ("Cargo mensual promedio por abandono", "Suma de cargos válidos / clientes con cargo válido, por grupo"),
]
