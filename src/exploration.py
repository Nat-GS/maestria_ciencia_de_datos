"""Indicadores y gráficos calculados sobre datos observados, sin Streamlit."""
import pandas as pd
import plotly.express as px

from src.data_processing import NUMERIC

LABELS = {
    "customerID": "Identificador", "gender": "Género", "SeniorCitizen": "Adulto mayor",
    "Partner": "Pareja", "Dependents": "Dependientes", "tenure": "Antigüedad (meses)",
    "PhoneService": "Servicio telefónico", "MultipleLines": "Varias líneas",
    "InternetService": "Servicio de internet", "OnlineSecurity": "Seguridad en línea",
    "OnlineBackup": "Respaldo en línea", "DeviceProtection": "Protección de dispositivos",
    "TechSupport": "Soporte técnico", "StreamingTV": "Streaming de televisión",
    "StreamingMovies": "Streaming de películas", "Contract": "Contrato",
    "PaperlessBilling": "Facturación electrónica", "PaymentMethod": "Método de pago",
    "MonthlyCharges": "Cargo mensual", "TotalCharges": "Cargo total", "Churn": "Abandono",
    "TramoAntiguedad": "Intervalo de antigüedad", "TramoCargo": "Intervalo de cargo mensual",
}
TRANSLATIONS = {
    "Yes": "Sí", "No": "No", "Female": "Femenino", "Male": "Masculino",
    "Month-to-month": "Mensual", "One year": "Un año", "Two year": "Dos años",
    "Fiber optic": "Fibra óptica", "DSL": "DSL", "No internet service": "Sin internet",
    "No phone service": "Sin servicio telefónico", "Electronic check": "Cheque electrónico",
    "Mailed check": "Cheque por correo", "Bank transfer (automatic)": "Transferencia automática",
    "Credit card (automatic)": "Tarjeta automática", "0": "No", "1": "Sí",
}
COLORS = ["#198580", "#e07849"]


def display_value(value):
    return TRANSLATIONS.get(str(value), str(value))


def add_segments(frame):
    result = frame.copy()
    result["TramoAntiguedad"] = pd.cut(result.tenure, [-1, 12, 24, 48, 72],
                                     labels=["0–12 meses", "13–24 meses", "25–48 meses", "49–72 meses"])
    result["TramoCargo"] = pd.cut(result.MonthlyCharges, [-float("inf"), 35, 70, 100, float("inf")],
                                labels=["Hasta 35", "Más de 35 a 70", "Más de 70 a 100", "Más de 100"])
    return result


def summary(frame):
    return {"clientes": len(frame), "abandonos": int(frame.Churn.eq("Yes").sum()),
            "tasa": float(frame.Churn.eq("Yes").mean() * 100),
            "cargo": float(frame.MonthlyCharges.mean()), "antiguedad": float(frame.tenure.mean())}


def abandonment_table(frame, column):
    work = add_segments(frame)
    table = work.groupby(column, observed=True, dropna=False).agg(
        Clientes=("Churn", "size"), Abandonos=("Churn", lambda s: s.eq("Yes").sum())
    ).reset_index()
    table["Abandono (%)"] = 100 * table.Abandonos / table.Clientes
    table[column] = table[column].astype("string").fillna("Sin dato").map(display_value)
    return table.rename(columns={column: LABELS.get(column, column)})


def abandonment_chart(frame, column):
    table = abandonment_table(frame, column)
    label = LABELS.get(column, column)
    fig = px.bar(table, x=label, y="Abandono (%)", text_auto=".1f",
                 hover_data=["Clientes", "Abandonos"], color_discrete_sequence=[COLORS[0]])
    fig.update_layout(yaxis_range=[0, 100], template="plotly_white", margin=dict(t=25, b=20))
    return fig


def distribution_chart(frame, column):
    work = frame.copy()
    work["Abandono"] = work.Churn.map({"No": "Permaneció", "Yes": "Abandonó"})
    if column not in NUMERIC:
        work[column] = work[column].map(display_value)
    fig = px.histogram(work, x=column, color="Abandono", barmode="group", nbins=30,
                       labels={column: LABELS[column]},
                       category_orders={"Abandono": ["Permaneció", "Abandonó"]},
                       color_discrete_sequence=COLORS)
    fig.update_layout(template="plotly_white", yaxis_title="Clientes", margin=dict(t=25, b=20))
    return fig


def monthly_by_churn(frame):
    table = frame.groupby("Churn").MonthlyCharges.agg(["count", "mean"]).reset_index()
    table.Churn = table.Churn.map({"No": "Permaneció", "Yes": "Abandonó"})
    return table.rename(columns={"Churn": "Abandono", "count": "Clientes con cargo válido",
                                 "mean": "Cargo mensual promedio"})


def data_dictionary():
    descriptions = {
        "customerID": "Código alfanumérico que identifica a cada cliente y permite detectar registros repetidos.",
        "gender": "Género registrado del cliente: femenino o masculino.",
        "SeniorCitizen": "Indica si el cliente está clasificado como adulto mayor: 1 significa sí y 0 significa no.",
        "Partner": "Indica si el cliente tiene pareja, según la información registrada.",
        "Dependents": "Indica si el cliente tiene personas dependientes a su cargo.",
        "tenure": "Cantidad de meses que el cliente ha permanecido en la empresa de telecomunicaciones.",
        "PhoneService": "Indica si el cliente tiene contratado el servicio de telefonía.",
        "MultipleLines": "Indica si el cliente dispone de varias líneas telefónicas o no tiene servicio de telefonía.",
        "InternetService": "Tipo de servicio de internet contratado: DSL, fibra óptica o sin servicio de internet.",
        "OnlineSecurity": "Indica si el cliente tiene seguridad en línea, no la tiene o no dispone de internet.",
        "OnlineBackup": "Indica si el cliente tiene respaldo en línea, no lo tiene o no dispone de internet.",
        "DeviceProtection": "Indica si el cliente tiene protección de dispositivos, no la tiene o no dispone de internet.",
        "TechSupport": "Indica si el cliente tiene soporte técnico contratado, no lo tiene o no dispone de internet.",
        "StreamingTV": "Indica si el cliente tiene servicio de televisión por streaming o no dispone de este servicio o de internet.",
        "StreamingMovies": "Indica si el cliente tiene servicio de películas por streaming o no dispone de este servicio o de internet.",
        "Contract": "Duración del contrato del cliente: mensual, de un año o de dos años.",
        "PaperlessBilling": "Indica si el cliente utiliza facturación electrónica, sin recibir una factura en papel.",
        "PaymentMethod": "Método de pago registrado: cheque electrónico, cheque por correo, transferencia automática o tarjeta automática.",
        "MonthlyCharges": "Importe mensual cobrado al cliente por sus servicios, expresado en unidades monetarias.",
        "TotalCharges": "Importe total acumulado cobrado al cliente durante su permanencia, expresado en unidades monetarias.",
        "Churn": "Indica si el cliente abandonó el servicio: Yes corresponde a abandono y No a permanencia.",
    }
    return pd.DataFrame([{
        "Variable": key, "Descripción": descriptions[key],
        "Tipo": "Numérica" if key in NUMERIC else "Categórica",
    } for key, label in LABELS.items() if key not in ["TramoAntiguedad", "TramoCargo"]])
