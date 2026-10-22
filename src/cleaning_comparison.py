"""Comparaciones descriptivas del pipeline ya ajustado; nunca aprende parámetros."""
import pandas as pd
import plotly.graph_objects as go

from src.cleaning import field_checks
from src.integration import CONTRACT_MAP
from src.interpretation import Exhibit, TEAL, CORAL, number, styled
from src.modeling import MODEL_NAMES


def _bars(table, category, series, y):
    fig = go.Figure()
    for name, color in zip(series, [CORAL, TEAL, "#8295a3"]):
        fig.add_bar(x=table[category], y=table[name], name=name, marker_color=color)
    fig.update_layout(barmode="group", height=380)
    return styled(fig, category, y)


def cleaning_comparison(raw, steps, preparation, experiment):
    """Devuelve gráficos/tablas calculados sin fit ni cambios en las entradas.

    Las etapas proceden de cleaning_steps y prepare_integrated. La comparación
    de contratos mantiene todas las filas para aislar la normalización del efecto
    de consolidar clientes; el resumen final sí compara las dos poblaciones.
    """
    initial, final = steps.initial, steps.cleaned
    extra = int(initial.customerID.duplicated().sum())
    exact = int(raw.duplicated().sum())
    records = pd.DataFrame({"Indicador": ["Registros", "Filas adicionales por ID", "Copias exactas"],
                            "Entrada inicial": [len(initial), extra, exact],
                            "Vista final": [len(final), int(final.customerID.duplicated().sum()),
                                            int(final.duplicated().sum())]})
    exhibits = [Exhibit("Registros y duplicados",
        _bars(records, "Indicador", ["Entrada inicial", "Vista final"], "Cantidad de filas"), records,
        f"La base pasa de {number(len(initial),0)} filas a {number(len(final),0)} clientes consolidados. "
        f"Las {number(exact,0)} copias exactas están incluidas en las {number(extra,0)} filas adicionales por ID; "
        "no se suman nuevamente. Se conserva la primera aparición por archivo y fila.")]

    variables = ["tenure", "MonthlyCharges", "TotalCharges", "Contract"]
    missing = pd.DataFrame({"Variable": variables,
        "Entrada inicial": initial[variables].isna().sum().to_numpy(),
        "Vista final": final[variables].isna().sum().to_numpy()})
    exhibits.append(Exhibit("Valores faltantes por variable",
        _bars(missing, "Variable", ["Entrada inicial", "Vista final"], "Celdas faltantes"), missing,
        "En la entrada: " + "; ".join(f"{r.Variable}: {number(r[1],0)}" for r in missing.itertuples(index=False)) +
        f". En esas variables quedan {number(missing['Vista final'].sum(),0)} faltantes en la vista final. "
        "Los números se completan con medianas y los contratos con la moda aprendidas solo en entrenamiento; "
        "son estimaciones, no valores recuperados del cliente."))

    stages = pd.DataFrame({"Etapa": ["Entrada", "Consolidación por ID", "Contratos y negativos",
                                     "Extremos marcados", "Vista final"],
        "Celdas faltantes": [int(f.isna().sum().sum()) for f in
            [initial, steps.deduplicated, preparation.frame, steps.marked, final]]})
    fig = go.Figure(go.Scatter(x=stages.Etapa, y=stages['Celdas faltantes'], mode="lines+markers+text",
        text=[number(n,0) for n in stages['Celdas faltantes']], textposition="top center",
        name="Celdas faltantes", line=dict(color=TEAL), showlegend=True))
    fig.update_layout(height=400)
    exhibits.append(Exhibit("Calidad por etapa", styled(fig,"Etapa de preparación","Celdas faltantes"), stages,
        "Evolución: " + " → ".join(number(n,0) for n in stages['Celdas faltantes']) +
        ". El aumento intermedio hace explícitos contratos no reconocidos, cargos negativos y extremos "
        "antes de completarlos. Las dos primeras etapas también reflejan la retirada de filas repetidas."))

    negative_initial = int(initial.MonthlyCharges.lt(0).sum())
    negative_unique = int(steps.deduplicated.MonthlyCharges.lt(0).sum())
    high = int(preparation.frame.MonthlyCharges.gt(steps.upper_bound).sum())
    median = experiment.models[MODEL_NAMES[0]]["preparacion"].named_transformers_["numericas"]["imputacion"].statistics_[1]
    explanation = (
        f"Se observan {number(negative_initial,0)} cargos negativos en la entrada: "
        f"{number(negative_initial-negative_unique,0)} quedan fuera al consolidar y "
        f"{number(negative_unique,0)} se marcan como faltantes. Sobre los clientes únicos, "
        f"{number(high,0)} cargos superan el límite entrenado de {number(steps.upper_bound,2)} u. m. "
        f"Estos valores se marcan como faltantes y se completan con la mediana de entrenamiento "
        f"({number(median,2)} u. m.), junto con los vacíos existentes. No se convierten negativos en positivos "
        "ni se eliminan clientes por sus cargos. La entrada incluye filas repetidas; la vista final contiene clientes únicos.")
    values = pd.concat([initial.MonthlyCharges, final.MonthlyCharges]).dropna()
    low, high_value = values.min(), values.max()
    width = (high_value-low)/25 if high_value > low else 1
    bins = dict(start=float(low), end=float(high_value+width), size=float(width))
    hist, box = go.Figure(), go.Figure()
    for name, frame, color in [("Entrada inicial", initial, CORAL), ("Vista final", final, TEAL)]:
        hist.add_trace(go.Histogram(x=frame.MonthlyCharges.dropna(), xbins=bins,
            name=name, marker_color=color, opacity=.65))
        box.add_trace(go.Box(x=frame.MonthlyCharges.dropna(), name=name, marker_color=color,
                            boxpoints="outliers", quartilemethod="linear", showlegend=True))
    for fig in [hist, box]:
        fig.add_vline(x=0, line_dash="dot", line_color="#8295a3")
        fig.add_vline(x=steps.upper_bound, line_dash="dash", line_color=TEAL,
                      annotation_text="Límite aprendido en entrenamiento")
        fig.update_layout(height=380)
    hist.update_layout(barmode="overlay")
    exhibits.append(Exhibit("Distribución de cargos mensuales antes y después",
        styled(hist,"Cargo mensual (u. m.)","Frecuencia (valores disponibles)"), pd.DataFrame(), explanation))
    exhibits.append(Exhibit("Diagramas de caja de cargos mensuales",
        styled(box,"Cargo mensual (u. m.)","Versión de los datos"), pd.DataFrame(),
        "Las cajas muestran mediana, cuartiles y puntos fuera de los bigotes de cada versión. "
        "Los bigotes son descriptivos y pueden diferir del límite fijo aprendido por el pipeline. "
        "La línea discontinua señala ese límite entrenado; la línea punteada identifica cero."))

    normalized = initial.Contract.astype("string").str.casefold().map(CONTRACT_MAP)
    contracts = pd.concat([initial.Contract.fillna("Sin dato").value_counts().rename("Antes de normalizar"),
        normalized.fillna("Sin dato").value_counts().rename("Después de normalizar")],axis=1).fillna(0).astype(int)
    contracts = contracts.rename_axis("Contrato").reset_index()
    actions = preparation.actions.set_index("Problema")["Filas o celdas"]
    contract_fig = _bars(contracts,"Contrato",["Antes de normalizar","Después de normalizar"],"Cantidad de filas")
    contract_fig.update_xaxes(tickangle=-30)
    contract_fig.update_layout(height=470)
    exhibits.append(Exhibit("Contratos antes y después de la normalización", contract_fig, contracts,
        f"Se corrigen {number(actions['Variantes de contrato'],0)} variantes y "
        f"{number(actions['Contrato no reconocido'],0)} contratos no reconocidos pasan a ausentes. "
        f"Ambas series conservan las {number(len(initial),0)} filas de entrada para aislar la normalización: "
        "este gráfico no elimina duplicados ni completa contratos. La vista final del resumen sí incluye esos pasos."))

    def summary(frame, original=False):
        return [len(frame),frame.customerID.nunique(),len(frame.columns),int(frame.isna().sum().sum()),
            int(raw.duplicated().sum()) if original else int(frame.duplicated().sum()),
            int(frame.customerID.duplicated().sum()),
            int(field_checks(frame,steps.upper_bound,completed=not original)['Valores incorrectos'].sum())]
    summary_table = pd.DataFrame({"Indicador": ["Registros","Clientes únicos","Columnas originales",
        "Celdas faltantes","Copias exactas","Filas adicionales por ID","Valores inválidos (incluye extremos mensuales)"],
        "Entrada inicial": summary(initial,True),"Vista final": summary(final)})
    predictor_count = len(experiment.models[MODEL_NAMES[0]]["preparacion"].get_feature_names_out())
    return exhibits, summary_table, predictor_count
