"""Carga, contrato del dataset y transformaciones deterministas sin aprendizaje."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PATH = ROOT / "data" / "telco.csv"
DATA_PATH = ROOT / "data" / "telco_dirty.csv"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORIES = {
    "gender": ["Female", "Male"], "SeniorCitizen": ["0", "1"],
    "Partner": ["No", "Yes"], "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{c: ["No", "Yes", "No internet service"] for c in [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"]},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"],
}
FEATURES = list(CATEGORIES) + NUMERIC
COLUMNS = ["customerID"] + FEATURES + ["Churn"]


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Conserva los valores originales, incluidos espacios vacíos, para auditarlos."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(COLUMNS)
    if missing or extra:
        raise ValueError(f"Esquema incompatible. Faltantes: {sorted(missing)}; adicionales: {sorted(extra)}")
    if frame.empty:
        raise ValueError("El dataset está vacío.")
    return frame


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos sin imputar, eliminar filas o aprender del conjunto completo."""
    frame = raw.copy()
    for col in frame:
        if pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_string_dtype(frame[col]):
            frame[col] = frame[col].str.strip().replace("", np.nan)
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def quality_report(raw: pd.DataFrame) -> pd.DataFrame:
    clean = clean_data(raw)
    records = []
    for col in COLUMNS:
        missing = int(clean[col].isna().sum())
        invalid = 0
        if col in NUMERIC:
            text = raw[col].astype("string").str.strip()
            invalid = int((clean[col].isna() & text.notna() & text.ne("")).sum())
        elif col in CATEGORIES or col == "Churn":
            domain = CATEGORIES.get(col, ["No", "Yes"])
            invalid = int((clean[col].notna() & ~clean[col].isin(domain)).sum())
        out_of_range = 0
        if col in NUMERIC:
            out_of_range = int((clean[col].lt(0) | ~np.isfinite(clean[col].fillna(0))).sum())
            if col == "tenure":
                out_of_range = int((clean[col].lt(0) | clean[col].gt(72) |
                                    clean[col].mod(1).ne(0) & clean[col].notna()).sum())
        records.append({"Variable": col, "Tipo original": str(raw[col].dtype),
                        "Tipo analítico": str(clean[col].dtype), "Ausentes": missing,
                        "Ausentes (%)": round(100 * missing / len(raw), 3),
                        "Valores inválidos": invalid, "Fuera de rango": out_of_range,
                        "Valores distintos": int(clean[col].nunique())})
    return pd.DataFrame(records)


def consistency_report(frame: pd.DataFrame) -> pd.DataFrame:
    issues = {
        "Identificadores repetidos": frame.customerID.duplicated().sum(),
        "Identificadores ausentes": frame.customerID.isna().sum(),
        "Filas duplicadas": frame.duplicated().sum(),
        "Líneas incompatibles con telefonía": (
            frame.PhoneService.eq("No").ne(frame.MultipleLines.eq("No phone service"))
        ).sum(),
    }
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies"]:
        issues[f"Internet incompatible con {col}"] = (
            frame.InternetService.eq("No").ne(frame[col].eq("No internet service"))
        ).sum()
    return pd.DataFrame([{"Comprobación": key, "Incidencias": int(value)} for key, value in issues.items()])


def validate_for_model(raw: pd.DataFrame) -> pd.DataFrame:
    # Las copias del notebook son idénticas, incluidos sus nulos. Se retiene la
    # primera fila antes de separar para impedir que un cliente cruce particiones.
    raw = raw.drop_duplicates(keep="first")
    report = quality_report(raw)
    if report[["Valores inválidos", "Fuera de rango"]].to_numpy().sum():
        raise ValueError("Hay valores inválidos o fuera de rango. Revise la calidad antes de modelar.")
    frame = clean_data(raw)
    if frame.Churn.isna().any() or frame.Churn.nunique() != 2:
        raise ValueError("Churn debe contener ambas clases y ninguna etiqueta ausente.")
    if consistency_report(frame).Incidencias.sum():
        raise ValueError("Hay inconsistencias o identificadores repetidos; revise el informe de calidad.")
    return frame


def dataset_comparison(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Auditoría comparable; no inventa anomalías ni usa límites aprendidos."""
    records = []
    for name, raw in datasets.items():
        frame = clean_data(raw)
        quality = quality_report(raw)
        consistency = consistency_report(frame)
        record = {
            "Versión": name, "Registros": len(frame), "Columnas": len(frame.columns),
            "Clientes únicos": int(frame.customerID.nunique()),
            "Duplicados exactos": int(frame.duplicated().sum()),
            "IDs repetidos": int(frame.customerID.duplicated().sum()),
            "Ausentes totales": int(frame.isna().sum().sum()),
            "Valores inválidos": int(quality["Valores inválidos"].sum()),
            "Fuera de rango": int(quality["Fuera de rango"].sum()),
            "Inconsistencias de servicios": int(consistency.iloc[3:].Incidencias.sum()),
            "Abandonos": int(frame.Churn.eq("Yes").sum()),
        }
        record.update({f"Ausentes {col}": int(frame[col].isna().sum()) for col in NUMERIC})
        records.append(record)
    return pd.DataFrame(records)


class MonthlyOutlierFilter(TransformerMixin, BaseEstimator):
    """Marca cargos mensuales extremos con un límite aprendido solo de entrenamiento."""
    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)[:, 1]
        values = values[np.isfinite(values) & (values >= 0)]
        if not len(values):
            raise ValueError("No hay cargos mensuales válidos para estimar el límite.")
        q1, q3 = np.quantile(values, [.25, .75])
        self.upper_bound_ = float(q3 + 1.5 * (q3 - q1))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float).copy()
        values[values[:, 1] > self.upper_bound_, 1] = np.nan
        return values

    def get_feature_names_out(self, input_features=None):
        return np.asarray(NUMERIC if input_features is None else input_features, dtype=object)


def make_preprocessor(scale: bool = True, handle_outliers: bool = False) -> ColumnTransformer:
    steps = []
    if handle_outliers:
        steps.append(("atipicos", MonthlyOutlierFilter()))
    steps.append(("imputacion", SimpleImputer(strategy="median")))
    if scale:
        steps.append(("escalado", StandardScaler()))
    categorical = Pipeline([
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("codificacion", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numericas", Pipeline(steps), NUMERIC),
        ("categoricas", categorical, list(CATEGORIES)),
    ], verbose_feature_names_out=False)
