"""Integración vertical y preparación determinista de los tres archivos entregados."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing import ROOT, CATEGORIES, NUMERIC, clean_data, load_data

SOURCE_PATHS = tuple(ROOT / "data" / f"telco_csv_{i}.csv" for i in (1, 2, 3))
CONTRACT_MAP = {
    "month-to-month": "Month-to-month", "month to month": "Month-to-month",
    "one year": "One year", "two year": "Two year", "two years": "Two year",
}


@dataclass
class IntegratedData:
    raw: pd.DataFrame
    provenance: pd.DataFrame
    inventory: pd.DataFrame


@dataclass
class Preparation:
    frame: pd.DataFrame
    actions: pd.DataFrame
    conflicts: pd.DataFrame


def load_project_data(paths=SOURCE_PATHS) -> IntegratedData:
    frames, provenance, inventory = [], [], []
    for path in paths:
        path = Path(path)
        data = load_data(path)
        frames.append(data)
        provenance.append(pd.DataFrame({"Archivo": path.name, "Fila CSV": np.arange(len(data)) + 2}))
        inventory.append({"Archivo": path.name, "Filas": len(data), "Columnas": len(data.columns),
                          "SHA-256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not frames:
        raise ValueError("Se necesita al menos un archivo de datos.")
    return IntegratedData(pd.concat(frames, ignore_index=True),
                          pd.concat(provenance, ignore_index=True), pd.DataFrame(inventory))


def prepare_integrated(raw: pd.DataFrame) -> Preparation:
    """Normalizar dominios, separar clientes y marcar errores; no imputa ni estima límites."""
    frame = clean_data(raw)
    actions = []
    def record(problem, number, treatment):
        actions.append({"Problema": problem, "Filas o celdas": int(number), "Tratamiento": treatment})
    contracts = frame.Contract.astype("string").str.casefold().map(CONTRACT_MAP)
    variants = frame.Contract.notna() & contracts.notna() & frame.Contract.ne(contracts)
    unknown = frame.Contract.notna() & contracts.isna()
    record("Variantes de contrato", variants.sum(), "Normalizar variantes inequívocas de escritura")
    record("Contrato no reconocido", unknown.sum(), "Convertir a ausente, sin asumir un contrato")
    frame["Contract"] = contracts.astype(object).where(contracts.notna(), np.nan)
    # Auditar antes de consolidar, con posiciones trazables a archivo/fila CSV.
    duplicate_ids = frame.customerID.duplicated(keep=False)
    conflicts = []
    for customer_id, group in frame.loc[duplicate_ids].groupby("customerID", sort=False):
        differing = [col for col in frame if col != "customerID" and group[col].nunique(dropna=False) > 1]
        if group.Churn.nunique(dropna=False) > 1:
            raise ValueError(f"Etiquetas de abandono contradictorias para {customer_id}; requiere revisión.")
        if differing:
            conflicts.append({"Cliente": customer_id, "Filas integradas": ", ".join(map(str, group.index)),
                              "Campos diferentes": ", ".join(differing),
                              "Fila conservada": int(group.index[0])})
    record("Copias exactas en la entrada", raw.duplicated().sum(), "Auditar; incluidas en la consolidación por ID")
    record("IDs con campos diferentes tras normalizar", len(conflicts),
           "Conservar primera ocurrencia por orden archivo 1→2→3 y fila; registrar conflicto")
    remove = frame.customerID.duplicated(keep="first")
    record("Filas adicionales del mismo cliente", remove.sum(),
           "Conservar primera ocurrencia antes de separar entrenamiento y prueba")
    frame = frame.loc[~remove].copy()
    for col in NUMERIC:
        invalid = ~np.isfinite(frame[col]) & frame[col].notna() | frame[col].lt(0)
        if col == "tenure":
            invalid |= frame[col].gt(72) | (frame[col].mod(1).ne(0) & frame[col].notna())
        record(f"Valores inválidos de {col}", invalid.sum(), "Convertir a ausente; imputar dentro del pipeline")
        frame.loc[invalid, col] = np.nan
    if frame.customerID.isna().any() or frame.Churn.isna().any():
        raise ValueError("Identificador o etiqueta ausente: requiere revisión antes de modelar.")
    # Mantener índices originales permite reconstruir la procedencia de cada fila.
    return Preparation(frame, pd.DataFrame(actions), pd.DataFrame(conflicts, columns=[
        "Cliente", "Filas integradas", "Campos diferentes", "Fila conservada"]))


def quality_summary(raw: pd.DataFrame) -> pd.DataFrame:
    from src.data_processing import quality_report, consistency_report
    frame = clean_data(raw)
    report = quality_report(raw)
    numeric = frame.MonthlyCharges.dropna()
    q1, q3 = numeric.quantile([.25, .75])
    upper = q3 + 1.5 * (q3 - q1)
    return pd.DataFrame([
        ("Registros", len(raw)), ("Clientes únicos", frame.customerID.nunique()),
        ("Copias exactas", raw.duplicated().sum()),
        ("Filas adicionales por ID", frame.customerID.duplicated().sum()),
        ("Clientes con campos diferentes en la entrada", frame.groupby("customerID").nunique(dropna=False).gt(1).any(axis=1).sum()),
        ("Celdas ausentes", frame.isna().sum().sum()),
        ("Categorías o textos inválidos", report["Valores inválidos"].sum()),
        ("Valores fuera de rango", report["Fuera de rango"].sum()),
        ("Cargos mensuales negativos", frame.MonthlyCharges.lt(0).sum()),
        ("Cargos superiores a Q3 + 1,5×RIC (descriptivo)", frame.MonthlyCharges.gt(upper).sum()),
        ("Inconsistencias de servicios", consistency_report(frame).iloc[3:].Incidencias.sum()),
    ], columns=["Control", "Cantidad"])
