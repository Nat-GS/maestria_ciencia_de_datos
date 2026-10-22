"""Comparación reproducible con validación interna y prueba final independiente."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from src.data_processing import CATEGORIES, FEATURES, make_preprocessor, validate_for_model

SEED = 42
MODEL_NAMES = ["Regresión logística", "Árbol de decisión", "Naive Bayes"]
BASELINE = "Referencia mayoritaria"


def experiment_signature():
    """Invalida entrenamientos cuando cambia el código de modelos o preparación."""
    import hashlib
    from pathlib import Path
    digest = hashlib.sha256()
    for name in ["modeling.py", "data_processing.py", "integration.py", "prediction.py"]:
        digest.update(Path(__file__).with_name(name).read_bytes())
    digest.update("|".join(MODEL_NAMES).encode())
    return digest.hexdigest()


def experiment_complete(experiment):
    expected = set(MODEL_NAMES)
    for name in ["models", "probabilities", "confusion", "parameters"]:
        if not expected.issubset(getattr(experiment, name, {})):
            return False
    for name in ["metrics", "validation"]:
        table = getattr(experiment, name, None)
        if table is None or "Modelo" not in table or not expected.issubset(table.Modelo):
            return False
    return True


@dataclass
class Experiment:
    models: dict
    metrics: pd.DataFrame
    validation: pd.DataFrame
    candidates: pd.DataFrame
    train_index: list
    test_index: list
    y_test: pd.Series
    probabilities: dict
    confusion: dict
    selected: str
    parameters: dict
    train_ids: list
    test_ids: list
    integrated: bool = False


def metric_values(y_true, probability):
    prediction = np.asarray(probability) >= 0.5
    return {"Accuracy": accuracy_score(y_true, prediction),
            "Precision": precision_score(y_true, prediction, zero_division=0),
            "Recall": recall_score(y_true, prediction, zero_division=0),
            "F1-score": f1_score(y_true, prediction, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, probability)}


def model_frame(raw: pd.DataFrame, integrated: bool = False):
    if integrated:
        from src.integration import prepare_integrated
        return validate_for_model(prepare_integrated(raw).frame)
    return validate_for_model(raw)


def run_experiment(raw: pd.DataFrame, *, integrated: bool = False, features=None) -> Experiment:
    frame = model_frame(raw, integrated)
    features = FEATURES if features is None else list(features)
    if not set(features).issubset(FEATURES) or not {"tenure", "MonthlyCharges"}.issubset(features):
        raise ValueError("El subconjunto debe contener antigüedad y cargo mensual, sin ID ni etiqueta.")
    x = frame[features]
    y = frame.Churn.map({"No": 0, "Yes": 1})
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    configs = {
        MODEL_NAMES[0]: (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"modelo__C": [0.1, 1.0, 10.0], "modelo__class_weight": [None, "balanced"]}, True),
        MODEL_NAMES[1]: (
            DecisionTreeClassifier(random_state=SEED),
            {"modelo__max_depth": [3, 5, 8], "modelo__min_samples_leaf": [20, 50],
             "modelo__class_weight": [None, "balanced"]}, False),
        MODEL_NAMES[2]: (
            GaussianNB(),
            {"modelo__var_smoothing": [1e-9, 1e-6, 1e-3]}, True),
    }
    models, parameters, validation, candidates = {}, {}, [], []
    for name, (estimator, grid, scale) in configs.items():
        preparation = make_preprocessor(scale, handle_outliers=integrated)
        # Reutilizar las mismas transformaciones; el filtro mensual requiere
        # tenure y MonthlyCharges en las dos primeras posiciones numéricas.
        preparation.transformers = [(key, transformer, [c for c in columns if c in features])
                                    for key, transformer, columns in preparation.transformers]
        pipeline = Pipeline([("preparacion", preparation), ("modelo", estimator)])
        search = GridSearchCV(pipeline, grid, cv=cv, refit="f1",
                              scoring={"f1": "f1", "recall": "recall", "roc_auc": "roc_auc"},
                              n_jobs=1, error_score="raise")
        search.fit(x_train, y_train)
        models[name] = search.best_estimator_
        parameters[name] = search.best_params_
        results, best = search.cv_results_, search.best_index_
        validation.append({"Modelo": name, "F1 medio (validación)": results["mean_test_f1"][best],
                           "Desviación F1": results["std_test_f1"][best],
                           "Recall medio (validación)": results["mean_test_recall"][best],
                           "ROC-AUC medio (validación)": results["mean_test_roc_auc"][best]})
        for index, params in enumerate(results["params"]):
            candidates.append({"Modelo": name, "Parámetros": str(params),
                               "F1 medio": results["mean_test_f1"][index],
                               "Recall medio": results["mean_test_recall"][index],
                               "ROC-AUC medio": results["mean_test_roc_auc"][index]})
    validation = pd.DataFrame(validation)
    selected = validation.sort_values(["F1 medio (validación)", "Recall medio (validación)",
                                      "ROC-AUC medio (validación)"], ascending=False).iloc[0].Modelo
    # La referencia usa el mismo entrenamiento y la misma prueba.
    models[BASELINE] = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    probabilities, matrices, metrics = {}, {}, []
    for name, model in models.items():
        probability = model.predict_proba(x_test)[:, 1]
        probabilities[name] = probability
        matrices[name] = confusion_matrix(y_test, probability >= 0.5, labels=[0, 1])
        metrics.append({"Modelo": name, **metric_values(y_test, probability)})
    return Experiment(models, pd.DataFrame(metrics), validation, pd.DataFrame(candidates),
                      x_train.index.tolist(), x_test.index.tolist(), y_test,
                      probabilities, matrices, selected, parameters,
                      frame.loc[x_train.index, "customerID"].tolist(),
                      frame.loc[x_test.index, "customerID"].tolist(), integrated)


def compare_experiments(original: Experiment, dirty: Experiment) -> pd.DataFrame:
    if original.train_ids != dirty.train_ids or original.test_ids != dirty.test_ids:
        raise ValueError("La comparación requiere los mismos clientes en entrenamiento y prueba.")
    comparison = original.metrics.merge(dirty.metrics, on="Modelo", suffixes=(" original", " sucio"))
    for metric in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
        comparison[f"Δ {metric}"] = comparison[f"{metric} sucio"] - comparison[f"{metric} original"]
    return comparison


def prepared_view(experiment: Experiment, raw: pd.DataFrame, name: str) -> pd.DataFrame:
    """Vista imputada sin codificar; usa solo estadísticas del modelo entrenado."""
    frame = model_frame(raw, experiment.integrated).copy()
    preprocessor = experiment.models[name]["preparacion"]
    numeric = preprocessor.named_transformers_["numericas"]
    from src.data_processing import NUMERIC
    values = frame[NUMERIC]
    if "atipicos" in numeric.named_steps:
        values = numeric["atipicos"].transform(values)
    frame[NUMERIC] = numeric["imputacion"].transform(values)
    frame[list(CATEGORIES)] = preprocessor.named_transformers_["categoricas"]["imputacion"].transform(frame[list(CATEGORIES)])
    frame["Partición"] = np.where(frame.index.isin(experiment.train_index), "Entrenamiento", "Prueba")
    return frame


def validate_customer(customer: dict) -> pd.DataFrame:
    missing = set(FEATURES) - set(customer)
    if missing:
        raise ValueError(f"Faltan características: {sorted(missing)}")
    for column, domain in CATEGORIES.items():
        if customer[column] not in domain:
            raise ValueError(f"Categoría inválida en {column}.")
    tenure, monthly, total = (float(customer[c]) for c in ["tenure", "MonthlyCharges", "TotalCharges"])
    if not np.isfinite([tenure, monthly, total]).all():
        raise ValueError("Los valores numéricos deben ser finitos.")
    if tenure < 0 or tenure > 72 or not tenure.is_integer() or monthly < 0 or total < 0:
        raise ValueError("Revise antigüedad y cargos: antigüedad entera entre 0 y 72, cargos no negativos.")
    if (customer["PhoneService"] == "No") != (customer["MultipleLines"] == "No phone service"):
        raise ValueError("Las líneas adicionales deben ser coherentes con el servicio telefónico.")
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        if (customer["InternetService"] == "No") != (customer[col] == "No internet service"):
            raise ValueError(f"{col} debe ser coherente con el servicio de internet.")
    return pd.DataFrame([customer])[FEATURES]


def predict_customer(model, customer: dict) -> float:
    return float(model.predict_proba(validate_customer(customer))[0, 1])


def importance_table(model):
    names = model["preparacion"].get_feature_names_out()
    estimator = model["modelo"]
    if isinstance(estimator, GaussianNB):
        return pd.DataFrame({"Variable transformada": names,
                             "Media si permanece": estimator.theta_[0],
                             "Media si abandona": estimator.theta_[1]})
    coefficient = hasattr(estimator, "coef_")
    values = estimator.coef_[0] if coefficient else estimator.feature_importances_
    result = pd.DataFrame({"Variable transformada": names, "Valor": values})
    return result.loc[result.Valor.abs().sort_values(ascending=False).index].reset_index(drop=True)


def roc_chart(experiment):
    fig = go.Figure()
    for name, probabilities in experiment.probabilities.items():
        fpr, tpr, _ = roc_curve(experiment.y_test, probabilities)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Azar", line=dict(dash="dash", color="#a0a0a0")))
    fig.update_layout(template="plotly_white", xaxis_title="Tasa de falsos positivos",
                      yaxis_title="Tasa de verdaderos positivos (recall)",
                      legend=dict(orientation="h", y=-0.25), margin=dict(t=20))
    return fig


def confusion_chart(matrix):
    fig = go.Figure(go.Heatmap(z=matrix, x=["Permanencia", "Abandono"],
                              y=["Permanencia", "Abandono"], text=matrix,
                              texttemplate="%{text}", colorscale="Teal", showscale=False))
    fig.update_layout(template="plotly_white", xaxis_title="Predicción", yaxis_title="Valor real",
                      yaxis_autorange="reversed", margin=dict(t=20))
    return fig
