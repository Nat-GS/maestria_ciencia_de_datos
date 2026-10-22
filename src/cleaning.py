"""Auditoría por etapas del mismo flujo usado por los modelos; sin Streamlit."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data_processing import CATEGORIES, NUMERIC, clean_data, consistency_report
from src.integration import prepare_integrated
from src.modeling import MODEL_NAMES, prepared_view
from src.interpretation import TEAL, CORAL, styled


@dataclass
class CleaningSteps:
    initial: pd.DataFrame
    deduplicated: pd.DataFrame
    marked: pd.DataFrame
    cleaned: pd.DataFrame
    upper_bound: float


def remove_duplicates(frame):
    # Una fila por cliente; también elimina las copias exactas.
    return frame.drop_duplicates(subset="customerID", keep="first").copy()


def cleaning_steps(raw, experiment):
    initial = clean_data(raw)
    deduplicated = remove_duplicates(initial)
    # Normalización de contratos y errores numéricos: misma función del modelo.
    marked = prepare_integrated(raw).frame.copy()
    numeric = experiment.models[MODEL_NAMES[0]]["preparacion"].named_transformers_["numericas"]
    # El límite procede solo del entrenamiento. Aquí no se calcula de nuevo.
    marked[NUMERIC] = numeric["atipicos"].transform(marked[NUMERIC])
    cleaned = prepared_view(experiment, raw, MODEL_NAMES[0]).drop(columns="Partición")
    return CleaningSteps(initial, deduplicated, marked, cleaned,
                         numeric["atipicos"].upper_bound_)


def field_checks(frame, upper_bound=None, completed=False):
    records = []
    for col in frame:
        values = frame[col]
        invalid = pd.Series(False, index=frame.index)
        if col in CATEGORIES or col == "Churn":
            invalid = values.notna() & ~values.isin(CATEGORIES.get(col, ["No", "Yes"]))
        elif col in NUMERIC:
            numeric = pd.to_numeric(values, errors="coerce")
            invalid = values.notna() & (numeric.isna() | ~np.isfinite(numeric) | numeric.lt(0))
            if col == "tenure":
                invalid |= numeric.gt(72)
                if not completed:
                    invalid |= numeric.notna() & numeric.mod(1).ne(0)
            if col == "MonthlyCharges" and upper_bound is not None:
                invalid |= numeric.gt(upper_bound)
        records.append({"Variable": col, "Nulos": int(values.isna().sum()),
                        "Valores incorrectos": int(invalid.sum())})
    return pd.DataFrame(records)


def final_checks(frame, upper_bound):
    fields = field_checks(frame, upper_bound, completed=True)
    return pd.DataFrame([
        ("Celdas nulas", int(fields.Nulos.sum())),
        ("Filas duplicadas completas", int(frame.duplicated().sum())),
        ("Clientes repetidos por identificador", int(frame.customerID.duplicated().sum())),
        ("Valores incorrectos según las reglas definidas", int(fields["Valores incorrectos"].sum())),
        ("Inconsistencias entre servicios", int(consistency_report(frame).iloc[3:].Incidencias.sum())),
    ], columns=["Comprobación", "Cantidad"])


def comparison_chart(labels, before, after, middle=None):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(labels), y=list(before), name="Antes", marker_color=CORAL))
    if middle is not None:
        fig.add_trace(go.Bar(x=list(labels), y=list(middle), name="Errores marcados como faltantes", marker_color="#9caeb8"))
    fig.add_trace(go.Bar(x=list(labels), y=list(after), name="Después", marker_color=TEAL))
    fig.update_layout(barmode="group", height=330)
    return styled(fig, None, "Cantidad")


def outlier_review(before, after):
    """Comparación descriptiva de extremos; no ajusta ni altera el pipeline."""
    from plotly.subplots import make_subplots
    from src.exploration import LABELS

    fig = make_subplots(rows=1, cols=3, subplot_titles=[LABELS[c] for c in NUMERIC])
    counts = []
    for index, col in enumerate(NUMERIC, start=1):
        record = {"Variable": LABELS[col]}
        for name, frame, color in [("Antes", before, CORAL), ("Después", after, TEAL)]:
            values = frame[col].dropna()
            q1, q3 = values.quantile([.25, .75])
            iqr = q3 - q1
            record[name] = int((values.lt(q1 - 1.5 * iqr) | values.gt(q3 + 1.5 * iqr)).sum())
            fig.add_trace(go.Box(
                y=values, name=name, legendgroup=name, showlegend=index == 1,
                marker_color=color, boxpoints="outliers", quartilemethod="linear",
                hovertemplate=f"{LABELS[col]}: %{{y:.2f}}<extra>{name}</extra>",
            ), row=1, col=index)
        counts.append(record)
    fig.update_layout(height=420, boxmode="group")
    return styled(fig), pd.DataFrame(counts)
