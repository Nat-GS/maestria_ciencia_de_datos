"""Gráficos e interpretaciones calculadas; sin dependencia de Streamlit."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data_processing import NUMERIC
from src.exploration import LABELS, display_value

TEAL, CORAL = "#237d78", "#c97855"


def number(value, digits=1):
    return f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def styled(fig, x=None, y=None):
    fig.update_layout(template="plotly_white", font=dict(family="Arial", color="#243747"),
                      margin=dict(l=10, r=10, t=15, b=15), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", xaxis_title=x, yaxis_title=y,
                      legend=dict(orientation="h", y=-.2))
    return fig


@dataclass
class Exhibit:
    title: str
    chart: go.Figure
    table: pd.DataFrame
    interpretation: str


def numeric_statistics(frame):
    return frame[NUMERIC].describe().T.rename(columns={
        "count": "Cantidad disponible", "mean": "Media", "std": "Desviación estándar",
        "min": "Mínimo", "25%": "Primer cuartil", "50%": "Mediana",
        "75%": "Tercer cuartil", "max": "Máximo"}).rename_axis("Variable").reset_index()


def descriptive_statistics(frame):
    """Estadísticos por fila; nulos excluidos y desviación estándar muestral."""
    return frame[NUMERIC].apply(pd.to_numeric, errors="coerce").describe().rename(index={
        "count": "Cantidad disponible", "mean": "Media", "std": "Desviación estándar",
        "min": "Mínimo", "25%": "Primer cuartil (25 %)", "50%": "Mediana (50 %)",
        "75%": "Tercer cuartil (75 %)", "max": "Máximo",
    }).rename_axis("Estadístico")


def spanish_statistics(table):
    result = table.astype(object).copy()
    for row in result.index:
        digits = 0 if row == "Cantidad disponible" else 2
        result.loc[row] = table.loc[row].map(
            lambda value: "Sin datos" if pd.isna(value) else number(value, digits))
    return result


def descriptive_interpretation(table):
    parts = ["Los estadísticos utilizan los valores disponibles de cada variable, excluyendo los ausentes; la desviación estándar es muestral."]
    mean = table.loc["Media", "tenure"]
    std = table.loc["Desviación estándar", "tenure"]
    if pd.notna(mean) and pd.notna(std):
        parts.append(f"La antigüedad media es {number(mean, 2)} meses, con una desviación de {number(std, 2)} meses que describe la dispersión de la permanencia.")
    mean = table.loc["Media", "TotalCharges"]
    median = table.loc["Mediana (50 %)", "TotalCharges"]
    if pd.notna(mean) and pd.notna(median):
        parts.append(f"La media del cargo total ({number(mean, 2)}) y su mediana ({number(median, 2)})" +
                     (" son compatibles con una distribución asimétrica hacia valores altos." if mean > median else " permiten comparar el promedio con el valor central de la distribución."))
    if table.loc["Mínimo", "MonthlyCharges"] < 0:
        parts.append("Los cargos mensuales negativos evidencian problemas de calidad que deben tratarse antes del modelado.")
    return " ".join(parts)


def frequency_table(frame, col):
    counts = frame[col].value_counts(dropna=False)
    return pd.DataFrame({"Categoría": ["Sin dato" if pd.isna(v) else display_value(v) for v in counts.index],
                         "Frecuencia": counts.to_numpy(),
                         "Porcentaje (%)": counts.to_numpy() / len(frame) * 100})


def numeric_exhibit(frame, col):
    values = frame[col].dropna()
    title = f"Distribución de {LABELS[col].lower()}"
    if values.empty:
        return Exhibit(title, styled(go.Figure()), pd.DataFrame(),
                       "No hay valores disponibles para representar esta variable o relacionarla con el abandono.")
    counts, edges = np.histogram(values, bins=25)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=counts, width=np.diff(edges)*.94,
                          marker_color=TEAL, name="Registros",
                          customdata=np.column_stack([edges[:-1], edges[1:]]),
                          hovertemplate="Intervalo: %{customdata[0]:.2f} a %{customdata[1]:.2f}<br>Registros: %{y}<extra></extra>"))
    median = values.median()
    lo = frame.loc[frame[col].le(median)]
    hi = frame.loc[frame[col].gt(median)]
    text = (f"El histograma agrupa {len(values):,} valores observados en 25 intervalos. "
            f"La mediana es {number(median, 2)}, la media {number(values.mean(), 2)} y el rango "
            f"{number(values.min(), 2)}–{number(values.max(), 2)}. Hay {frame[col].isna().sum():,} ausentes. ")
    if len(lo) and len(hi):
        text += (f"El abandono es {number(100*lo.Churn.eq('Yes').mean())} % en los {len(lo):,} registros "
                 f"con valores hasta la mediana y {number(100*hi.Churn.eq('Yes').mean())} % en los "
                 f"{len(hi):,} superiores. Esta comparación descriptiva no demuestra causalidad.")
    else:
        text += "No hay dos grupos con observaciones para comparar abandono respecto a la mediana."
    table = pd.DataFrame({"Desde": edges[:-1], "Hasta": edges[1:], "Registros": counts})
    return Exhibit(title, styled(fig, LABELS[col], "Cantidad de registros"), table, text)


def category_exhibit(frame, col):
    table = frequency_table(frame, col)
    observed = table.loc[table.Categoría.ne("Sin dato")]
    fig = go.Figure(go.Bar(x=observed.Frecuencia, y=observed.Categoría, orientation="h",
                          marker_color=TEAL, name="Registros"))
    fig.update_yaxes(autorange="reversed")
    top = table.iloc[0]
    text = (f"Las barras muestran la frecuencia de {LABELS[col].lower()}. La categoría más frecuente es "
            f"«{top['Categoría']}»: {int(top['Frecuencia']):,} registros ({number(top['Porcentaje (%)'])} %). ")
    missing = int(frame[col].isna().sum())
    if missing:
        text += f"Se omiten {missing} ausentes de las barras; están incluidos en la tabla y su denominador. "
    if col == "Churn":
        positives = frame.Churn.eq("Yes").sum()
        text += (f"Hay {positives:,} abandonos de {len(frame):,} registros "
                 f"({number(100*positives/len(frame))} %). El desbalance justifica revisar recall y F1 además de exactitud.")
    else:
        rates = frame.groupby(col, observed=True).Churn.agg(
            cantidad="size", tasa=lambda x: x.eq("Yes").mean()*100)
        if not rates.empty:
            highest = rates.tasa.idxmax()
            text += (f"La mayor tasa de abandono corresponde a «{display_value(highest)}»: "
                     f"{number(rates.loc[highest, 'tasa'])} % sobre {int(rates.loc[highest, 'cantidad']):,} registros. "
                     "Es una asociación; el tamaño de cada grupo importa para interpretarla.")
    return Exhibit(f"Distribución de {LABELS[col].lower()}", styled(fig, "Cantidad de registros", None), table, text)


def contract_exhibit(frame):
    # Igual que pd.crosstab del Colab: los contratos ausentes se excluyen.
    table = pd.crosstab(frame.Contract, frame.Churn).reindex(columns=["No", "Yes"], fill_value=0)
    table.columns = ["Permanecieron", "Abandonaron"]
    table["Total"] = table.sum(axis=1)
    table["Abandono (%)"] = table.Abandonaron / table.Total * 100
    table = table.reset_index().rename(columns={"Contract": "Contrato"})
    table.Contrato = table.Contrato.map(display_value)
    fig = go.Figure(go.Bar(x=table.Contrato, y=table["Abandono (%)"],
                          marker_color=CORAL, name="Abandono", text=table["Abandono (%)"].round(1),
                          customdata=table[["Total", "Abandonaron"]],
                          hovertemplate="%{x}<br>Abandono: %{y:.1f}%<br>Registros: %{customdata[0]}<br>Abandonos: %{customdata[1]}<extra></extra>"))
    fig.update_yaxes(range=[0, 100])
    if table.empty:
        text = "No hay contratos observados en esta selección para comparar tasas de abandono."
    else:
        highest, lowest = table.loc[table["Abandono (%)"].idxmax()], table.loc[table["Abandono (%)"].idxmin()]
        text = (f"La altura representa abandonos divididos por registros de cada contrato. "
                f"«{highest.Contrato}» presenta la mayor tasa: {number(highest['Abandono (%)'])} % "
                f"({int(highest.Abandonaron)} de {int(highest.Total)}); «{lowest.Contrato}», "
                f"la menor: {number(lowest['Abandono (%)'])} % ({int(lowest.Abandonaron)} de {int(lowest.Total)}). "
                f"Se excluyen {frame.Contract.isna().sum()} contratos ausentes, como en el notebook. "
                "En la vista sucia las variantes siguen separadas; no implica causalidad.")
    return Exhibit("Abandono por tipo de contrato", styled(fig, "Tipo de contrato", "Abandono (%)"), table, text)


def missing_exhibit(frame):
    table = frame.isna().sum().rename_axis("Variable").reset_index(name="Ausentes")
    affected = table.loc[table.Ausentes.gt(0)].sort_values("Ausentes", ascending=False)
    labels = affected.Variable.map(LABELS)
    fig = go.Figure(go.Bar(x=labels, y=affected.Ausentes, marker_color=CORAL))
    text = (f"Se cuentan {int(table.Ausentes.sum()):,} celdas ausentes en {len(affected)} variables. ")
    if not affected.empty:
        row = affected.iloc[0]
        text += f"{LABELS.get(row.Variable, row.Variable)} concentra {int(row.Ausentes)} ausentes. "
    text += "La imputación se ajusta con entrenamiento; no se eliminan clientes solo por tener valores faltantes."
    missing_rows = frame.isna().any(axis=1)
    if missing_rows.any() and (~missing_rows).any():
        text += (f" El abandono observado es {number(frame.loc[missing_rows].Churn.eq('Yes').mean()*100)} % "
                 f"en filas con ausentes y {number(frame.loc[~missing_rows].Churn.eq('Yes').mean()*100)} % "
                 "en filas completas; la diferencia es descriptiva.")
    return Exhibit("Dónde falta información", styled(fig, None, "Celdas ausentes"), table, text)


def cleaning_exhibit(before, after, col):
    fig = go.Figure()
    for label, frame, color in [("Antes", before, CORAL), ("Después", after, TEAL)]:
        fig.add_trace(go.Box(y=frame[col], name=label, marker_color=color, boxpoints="outliers"))
    text = (f"Los diagramas comparan mediana, cuartiles y extensión de {LABELS[col].lower()}. "
            f"Antes: {len(before):,} filas y {int(before[col].isna().sum())} ausentes; "
            f"después: {len(after):,} clientes y {int(after[col].isna().sum())} ausentes. "
            f"La mediana pasa de {number(before[col].median(), 2)} a {number(after[col].median(), 2)}. "
            f"El abandono por fila pasa de {number(before.Churn.eq('Yes').mean()*100)} % a "
            f"{number(after.Churn.eq('Yes').mean()*100)} % tras consolidar clientes. "
            "El cambio combina consolidación, tratamiento de errores e imputación; no prueba una mejora causal en retención.")
    return Exhibit(f"Antes y después · {LABELS[col].lower()}", styled(fig, None, LABELS[col]), pd.DataFrame(), text)


def model_conclusions(experiment):
    chosen = experiment.selected
    row = experiment.metrics.set_index("Modelo").loc[chosen]
    tn, fp, fn, tp = experiment.confusion[chosen].ravel()
    return [
        f"La validación cruzada selecciona {chosen.lower()} por F1 medio, sin usar la prueba para elegir.",
        f"En prueba identifica {tp} de {tp+fn} abandonos: recall {number(row.Recall*100, 2)} % y F1 {number(row['F1-score'], 4)}.",
        f"Genera {fp} falsas alertas y omite {fn} abandonos. La precisión de las alertas es {number(row.Precision*100, 2)} %; no equivale a retención garantizada.",
        f"Su ROC-AUC es {number(row['ROC-AUC'], 4)}. Las métricas corresponden a una partición aleatoria y probabilidades sin calibración operativa.",
    ]
