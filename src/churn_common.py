"""Funciones compartidas por el notebook y la app de Streamlit.

Mantener aquí la limpieza y el esquema garantiza que el modelo reciba en
producción exactamente las mismas columnas con las que se entrenó.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ARTIFACTS = ROOT / "artifacts"

ID_COL = "customerID"
TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL

# Nombres legibles para personas sin conocimientos técnicos.
LABELS = {
    "tenure": "Antigüedad (meses)",
    "MonthlyCharges": "Cargo mensual",
    "TotalCharges": "Total facturado",
    "gender": "Género",
    "SeniorCitizen": "Adulto mayor",
    "Partner": "Tiene pareja",
    "Dependents": "Tiene dependientes",
    "PhoneService": "Servicio telefónico",
    "MultipleLines": "Varias líneas",
    "InternetService": "Tipo de internet",
    "OnlineSecurity": "Seguridad en línea",
    "OnlineBackup": "Respaldo en línea",
    "DeviceProtection": "Protección de dispositivos",
    "TechSupport": "Soporte técnico",
    "StreamingTV": "TV por streaming",
    "StreamingMovies": "Películas por streaming",
    "Contract": "Tipo de contrato",
    "PaperlessBilling": "Factura electrónica",
    "PaymentMethod": "Método de pago",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza fila a fila: no aprende nada de los datos, así que no filtra información.

    Los `TotalCharges` en blanco quedan como NaN y se resuelven dentro del pipeline.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    if pd.api.types.is_numeric_dtype(out["SeniorCitizen"]):
        out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return out


def load_clean(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return clean(pd.read_csv(path))


def build_preprocessor() -> ColumnTransformer:
    """Un `TotalCharges` vacío corresponde a un cliente sin facturas, por eso se imputa 0."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def original_feature(transformed_name: str) -> str:
    """Devuelve la variable original de un nombre como `cat__Contract_One year`."""
    name = transformed_name.split("__", 1)[1]
    if name in NUMERIC:
        return name
    for col in CATEGORICAL:
        if name.startswith(col + "_"):
            return col
    raise ValueError(f"Variable no reconocida: {transformed_name}")


def group_by_original(values: np.ndarray, transformed_names) -> pd.Series:
    """Suma contribuciones de cada columna one-hot en su variable original."""
    groups = [original_feature(n) for n in transformed_names]
    return pd.Series(values, index=groups).groupby(level=0).sum()


# ---------------------------------------------------------------- calibración
def to_logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate(calibrator, p_raw) -> np.ndarray:
    """Convierte la salida del bosque en una probabilidad realista (Platt)."""
    return calibrator.predict_proba(to_logit(p_raw))[:, 1]


# ----------------------------------------------------------------------- LIME
def lime_category_maps(df: pd.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}


def lime_encode(df: pd.DataFrame, maps: dict[str, list[str]]) -> np.ndarray:
    """LIME trabaja con números: cada categoría pasa a su posición en `maps`."""
    out = df[FEATURES].copy()
    out["TotalCharges"] = out["TotalCharges"].fillna(0)
    for col, levels in maps.items():
        out[col] = out[col].map({v: i for i, v in enumerate(levels)})
    return out.to_numpy(dtype=float)


def lime_decode(arr: np.ndarray, maps: dict[str, list[str]]) -> pd.DataFrame:
    out = pd.DataFrame(arr, columns=FEATURES)
    for col, levels in maps.items():
        idx = out[col].round().astype(int).clip(0, len(levels) - 1)
        out[col] = np.asarray(levels, dtype=object)[idx]
    return out


def lime_label(text: str) -> str:
    """Traduce el nombre de variable de una regla de LIME, como `tenure <= 9.00`."""
    for feature in sorted(FEATURES, key=len, reverse=True):
        if feature in text:
            return text.replace(feature, LABELS[feature])
    return text


# -------------------------------------------------------------------- negocio
def plan_campaign(
    scored: pd.DataFrame,
    budget: float,
    offer_cost: float,
    success_rate: float,
    horizon_months: int,
    max_share: float = 1.0,
) -> pd.DataFrame:
    """Ordena clientes por valor esperado de contactarlos y aplica los límites.

    `scored` necesita `p_churn` y `MonthlyCharges`. Se contacta a un cliente solo si
    el ingreso esperado que se salva supera el coste de la oferta, y mientras
    alcancen el presupuesto y el porcentaje máximo de cartera.
    """
    plan = scored.copy()
    plan["ingreso_en_riesgo"] = plan["p_churn"] * plan["MonthlyCharges"] * horizon_months
    plan["ingreso_salvado_esperado"] = plan["ingreso_en_riesgo"] * success_rate
    plan["valor_neto_esperado"] = plan["ingreso_salvado_esperado"] - offer_cost
    plan = plan.sort_values("valor_neto_esperado", ascending=False).reset_index(drop=True)

    by_budget = int(budget // offer_cost) if offer_cost > 0 else len(plan)
    by_share = int(np.floor(len(plan) * max_share))
    positive = int((plan["valor_neto_esperado"] > 0).sum())
    plan["contactar"] = plan.index < min(by_budget, by_share, positive)
    return plan


def campaign_summary(plan: pd.DataFrame, offer_cost: float, subscription: float = 0.0) -> dict:
    chosen = plan[plan["contactar"]]
    spend = len(chosen) * offer_cost
    saved = float(chosen["ingreso_salvado_esperado"].sum())
    total_cost = spend + subscription
    return {
        "contactados": int(len(chosen)),
        "gasto_ofertas": float(spend),
        "ingreso_salvado": saved,
        "beneficio_neto": saved - total_cost,
        "roi": (saved - total_cost) / total_cost if total_cost > 0 else 0.0,
    }


def risk_band(p: float, threshold: float) -> str:
    if p >= max(threshold * 1.5, 0.5):
        return "Alto"
    if p >= threshold:
        return "Medio"
    return "Bajo"
