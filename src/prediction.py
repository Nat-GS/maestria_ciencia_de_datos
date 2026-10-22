"""Selección con entrenamiento y predicción reducida; sin dependencias de Streamlit."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.data_processing import CATEGORIES, FEATURES, NUMERIC
from src.modeling import MODEL_NAMES, SEED, model_frame, run_experiment

RISK_THRESHOLDS = (.30, .60)
TECHNIQUES = {
    "Regresión logística": "Estima la probabilidad de abandono mediante una combinación de predictores y una función logística. Permite examinar asociaciones mediante coeficientes.",
    "Árbol de decisión": "Divide los clientes mediante reglas sucesivas. Representa relaciones no lineales e interacciones con una estructura interpretable.",
    "Naive Bayes": "Combina probabilidades por clase suponiendo independencia condicional. GaussianNB utiliza distribuciones gaussianas también para los indicadores codificados; es una aproximación que debe contrastarse con los otros modelos.",
}


def training_importance(raw, experiment):
    """Importancia por permutación en reserva interna del entrenamiento.

    No accede a resultados ni etiquetas de la prueba final. Las configuraciones
    provienen de la búsqueda interna del entrenamiento completo, por lo que esta
    reserva sirve para explicar/seleccionar variables, no como evaluación final.
    """
    frame = model_frame(raw, True).loc[experiment.train_index]
    x, y = frame[FEATURES], frame.Churn.map({"No": 0, "Yes": 1})
    fit_index, validation_index = train_test_split(frame.index, test_size=.2, stratify=y, random_state=SEED)
    rows = []
    for name in MODEL_NAMES:
        model = clone(experiment.models[name]).fit(x.loc[fit_index], y.loc[fit_index])
        result = permutation_importance(model, x.loc[validation_index], y.loc[validation_index],
                                        scoring="f1", n_repeats=5, random_state=SEED, n_jobs=1)
        for col, mean, std in zip(FEATURES, result.importances_mean, result.importances_std):
            rows.append({"Modelo": name, "Variable": col, "Caída de F1": mean, "Desviación": std})
    return pd.DataFrame(rows)


@dataclass
class ReducedExperiment:
    experiment: object
    features: list
    ranking: pd.DataFrame


def run_reduced_experiment(raw, full_experiment, importance):
    ranking = importance.groupby("Variable", sort=False)["Caída de F1"].mean().sort_values(ascending=False)
    # Dos entradas numéricas base preservan el tratamiento mensual validado.
    required = ["tenure", "MonthlyCharges"]
    selected = required + [c for c in ranking.index if c not in required][:6]
    features = [c for c in FEATURES if c in selected]
    result = run_experiment(raw, integrated=True, features=features)
    if result.train_ids != full_experiment.train_ids or result.test_ids != full_experiment.test_ids:
        raise ValueError("El experimento reducido debe utilizar los mismos clientes y particiones.")
    table = ranking.rename("Caída media de F1").reset_index()
    table["Seleccionada"] = table.Variable.isin(features)
    table["Criterio"] = np.where(table.Variable.isin(required), "Entrada numérica base", "Orden de importancia")
    return ReducedExperiment(result, features, table)


def validate_reduced_customer(customer, features):
    if set(customer) != set(features):
        raise ValueError("Complete las variables del modelo simplificado.")
    for col in features:
        if col in CATEGORIES and customer[col] not in CATEGORIES[col]:
            raise ValueError(f"Categoría inválida: {col}.")
        if col in NUMERIC:
            value = float(customer[col])
            if not np.isfinite(value) or value < 0:
                raise ValueError("Los cargos y la antigüedad deben ser finitos y no negativos.")
            if col == "tenure" and (value > 72 or not value.is_integer()):
                raise ValueError("La antigüedad debe ser entera entre 0 y 72 meses.")
    internet = customer.get("InternetService")
    for col in ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]:
        if col in customer and internet is not None:
            if (internet == "No") != (customer[col] == "No internet service"):
                raise ValueError("Los servicios adicionales deben ser coherentes con el servicio de internet.")
    if "PhoneService" in customer and "MultipleLines" in customer:
        if (customer['PhoneService'] == 'No') != (customer['MultipleLines'] == 'No phone service'):
            raise ValueError("Las líneas deben ser coherentes con el servicio telefónico.")
    return pd.DataFrame([customer])[features]


def risk_level(probability):
    if probability < RISK_THRESHOLDS[0]:
        return "Bajo"
    if probability < RISK_THRESHOLDS[1]:
        return "Medio"
    return "Alto"


def profile_associations(raw, experiment, customer):
    """Comparaciones descriptivas con grupos observados SOLO de entrenamiento.

    No son contribuciones locales del modelo ni explicaciones causales.
    """
    frame = model_frame(raw, True).loc[experiment.train_index]
    target = frame.Churn.eq("Yes")
    baseline = float(target.mean())
    rows = []
    model = experiment.models[experiment.selected]
    bound = model['preparacion'].named_transformers_['numericas']['atipicos'].upper_bound_
    for col, value in customer.items():
        if col in NUMERIC:
            values = frame[col].where(frame[col].le(bound)) if col == 'MonthlyCharges' else frame[col]
            if col == 'MonthlyCharges' and value > bound:
                continue
            median = values.median()
            upper = value > median
            group = values.gt(median) if upper else values.le(median)
            label = f"{'Mayor' if upper else 'Menor o igual'} que la mediana ({median:.2f})"
        else:
            group = frame[col].eq(value)
            label = value
        if group.any():
            rate = float(target[group].mean())
            rows.append({"Variable": col, "Grupo comparable": label, "Clientes de entrenamiento": int(group.sum()),
                         "Abandono del grupo (%)": rate*100, "Diferencia frente al entrenamiento (pp)": (rate-baseline)*100})
    return pd.DataFrame(rows).sort_values('Diferencia frente al entrenamiento (pp)', ascending=False)


def retention_suggestions(customer):
    advice = []
    if customer.get("Contract") == "Month-to-month":
        advice.append("Ofrecer alternativas de contrato y explicar condiciones y beneficios sin compromisos forzados.")
    if customer.get("tenure", 73) <= 12:
        advice.append("Reforzar el acompañamiento inicial y consultar dificultades con el servicio.")
    if customer.get("TechSupport") == "No":
        advice.append("Revisar incidencias y facilitar acceso a soporte técnico.")
    if customer.get("OnlineSecurity") == "No":
        advice.append("Informar sobre opciones de seguridad adecuadas a las necesidades del cliente.")
    advice.append("Consultar satisfacción, revisar adecuación del plan y validar cualquier campaña con resultados observados.")
    return advice
