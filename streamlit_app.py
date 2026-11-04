"""Ancla: priorización de retención para ISP regionales.

Esta app solo carga los modelos exportados por churn_ml.ipynb; no reentrena nada.
"""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

from src.churn_common import (ARTIFACTS, CATEGORICAL, FEATURES, ID_COL, LABELS, RANDOM_STATE,
                              calibrate, campaign_summary, clean, group_by_original,
                              lime_decode, lime_encode, lime_label, load_clean, original_feature, plan_campaign,
                              risk_band)

st.set_page_config(page_title="Ancla | Retención para ISP", layout="wide")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MUTED, NEUTRAL = "#52514e", "#c3c2b7"
CHURN_COLORS = {"Se queda": BLUE, "Abandona": ORANGE}
SECTIONS = ["Exploración de datos", "Desempeño del modelo", "Predicción y explicación", "Aplicación de negocio"]


# ------------------------------------------------------------------ carga
@st.cache_resource
def load_models():
    pipeline = joblib.load(ARTIFACTS / "rf_final_pipeline.joblib")
    return {
        "final": pipeline,
        "calibrator": joblib.load(ARTIFACTS / "calibrator.joblib"),
        "logreg": joblib.load(ARTIFACTS / "logreg_pipeline.joblib"),
        "catboost": joblib.load(ARTIFACTS / "catboost_pipeline.joblib"),
        "explainer": shap.TreeExplainer(pipeline.named_steps["model"]),
        "names": pipeline.named_steps["prep"].get_feature_names_out(),
    }


@st.cache_resource
def load_lime():
    data = joblib.load(ARTIFACTS / "lime_background.joblib")
    maps = data["category_maps"]
    explainer = LimeTabularExplainer(
        data["background"], mode="classification", feature_names=FEATURES,
        class_names=["No", "Yes"], categorical_features=[FEATURES.index(c) for c in CATEGORICAL],
        categorical_names={FEATURES.index(c): maps[c] for c in CATEGORICAL},
        discretize_continuous=True, random_state=RANDOM_STATE)
    return explainer, maps


@st.cache_data
def load_tables():
    return {
        "models": pd.read_csv(ARTIFACTS / "model_comparison.csv"),
        "optimizers": pd.read_csv(ARTIFACTS / "optimizers.csv"),
        "pareto": pd.read_csv(ARTIFACTS / "pareto_trials.csv"),
        "best": pd.read_csv(ARTIFACTS / "best_so_far.csv"),
        "deciles": pd.read_csv(ARTIFACTS / "deciles.csv"),
        "scored": pd.read_csv(ARTIFACTS / "test_scored.csv"),
        "importance": pd.read_csv(ARTIFACTS / "importance_comparison.csv"),
    }


@st.cache_data
def load_metadata():
    return json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))


@st.cache_data
def load_eda():
    data = load_clean()
    data["Estado"] = np.where(data["Churn"] == "Yes", "Abandona", "Se queda")
    return data


def tidy(fig, height=360):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=44, b=10),
                      legend=dict(orientation="h", y=1.08, x=1, xanchor="right", title_text=""),
                      font=dict(color=MUTED), title_font=dict(size=15))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


def money(value):
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def percent(value):
    return f"{value:.1%}"


def literal_dollars(text):
    """Evita que Streamlit lea el texto entre dos signos $ como una fórmula."""
    return text.replace("$", "\\$")


# ------------------------------------------------------ A. Exploración
def section_eda():
    data = load_eda()
    st.header("Exploración de datos")
    st.caption("Datos reales del dataset Telco Customer Churn, una compañía ficticia de EE. UU. publicada por IBM.")

    c1, c2, c3, c4 = st.columns(4)
    contracts = c1.multiselect("Contrato", sorted(data["Contract"].unique()), default=sorted(data["Contract"].unique()))
    internet = c2.multiselect("Internet", sorted(data["InternetService"].unique()), default=sorted(data["InternetService"].unique()))
    payments = c3.multiselect("Método de pago", sorted(data["PaymentMethod"].unique()), default=sorted(data["PaymentMethod"].unique()))
    tenure = c4.slider("Antigüedad (meses)", 0, int(data["tenure"].max()), (0, int(data["tenure"].max())))

    view = data[data["Contract"].isin(contracts) & data["InternetService"].isin(internet)
                & data["PaymentMethod"].isin(payments) & data["tenure"].between(*tenure)]
    if view.empty:
        st.warning("Ningún cliente cumple estos filtros. Amplía la selección.")
        return

    churned = view[view["Estado"] == "Abandona"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clientes", f"{len(view):,}")
    k2.metric("Tasa de abandono", percent(len(churned) / len(view)),
              f"{(len(churned) / len(view) - (data['Estado'] == 'Abandona').mean()) * 100:+.1f} pp vs. total",
              delta_color="inverse")
    k3.metric("Ingreso mensual perdido", money(churned["MonthlyCharges"].sum()))
    k4.metric("Antigüedad mediana", f"{view['tenure'].median():.0f} meses")

    options = {LABELS[c]: c for c in ["Contract", "InternetService", "PaymentMethod", "TechSupport",
                                      "OnlineSecurity", "PaperlessBilling", "SeniorCitizen", "Dependents"]}
    dimension = options[st.selectbox("Comparar tasa de abandono por", list(options))]

    left, right = st.columns(2)
    rate = (view.groupby(dimension)["Estado"].apply(lambda s: (s == "Abandona").mean())
            .sort_values().reset_index(name="tasa"))
    fig = px.bar(rate, x="tasa", y=dimension, orientation="h", text=rate["tasa"].map("{:.0%}".format),
                 title=f"Tasa de abandono por {LABELS[dimension].lower()}",
                 labels={"tasa": "Tasa de abandono", dimension: ""}, color_discrete_sequence=[ORANGE])
    fig.update_xaxes(tickformat=".0%", rangemode="tozero")
    left_fig = tidy(fig)
    left_fig.update_traces(width=0.55)
    with left:
        show(left_fig)

    fig = px.histogram(view, x="tenure", color="Estado", nbins=24, barmode="stack",
                       color_discrete_map=CHURN_COLORS, title="Clientes por antigüedad",
                       labels={"tenure": "Antigüedad (meses)", "count": "Clientes"})
    fig.update_yaxes(title="Clientes")
    with right:
        show(tidy(fig))

    left, right = st.columns(2)
    fig = px.box(view, x="Estado", y="MonthlyCharges", color="Estado", color_discrete_map=CHURN_COLORS,
                 title="Cargo mensual según el resultado", labels={"MonthlyCharges": "Cargo mensual ($)", "Estado": ""})
    fig.update_layout(showlegend=False)
    with left:
        show(tidy(fig))

    buckets = pd.cut(view["tenure"], [-1, 6, 12, 24, 48, 72], labels=["0–6", "7–12", "13–24", "25–48", "49–72"])
    heat = (view.assign(Antigüedad=buckets).groupby(["Contract", "Antigüedad"], observed=True)["Estado"]
            .apply(lambda s: (s == "Abandona").mean()).unstack())
    fig = px.imshow(heat, text_auto=".0%", color_continuous_scale="Oranges", aspect="auto", zmin=0, zmax=0.7,
                    title="Tasa de abandono: contrato y antigüedad en meses",
                    labels={"x": "Antigüedad (meses)", "y": "", "color": "Tasa"})
    with right:
        show(tidy(fig))


# ---------------------------------------------- B. Desempeño del modelo
def section_performance():
    tables, meta = load_tables(), load_metadata()
    st.header("Desempeño del modelo")
    params = meta["selected_params"]
    st.caption(f"Modelo final: Random Forest con {params['n_estimators']} árboles, profundidad {params['max_depth']}, "
               f"mínimo {params['min_samples_leaf']} clientes por hoja y max_features {params['max_features']}.")

    st.subheader("Resultado final")
    st.caption("El conjunto de prueba se evaluó una sola vez, con modelo, umbral y calibración ya fijados.")
    cols = st.columns(5)
    names = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}
    for col, (key, label) in zip(cols, names.items()):
        col.metric(label, f"{meta['test_metrics'][key]:.3f}",
                   f"{meta['test_metrics'][key] - meta['cv_metrics'][key]:+.3f} vs. CV", delta_color="off")

    st.subheader("Comparación de modelos")
    st.caption("Validación cruzada de 5 folds sobre train, con los mismos folds para todos. Modelos con parámetros "
               "iniciales: el Random Forest optimizado alcanza mejores cifras, aunque CatBoost conserva un F1 algo mayor.")
    models = tables["models"].sort_values("f1", ascending=False)
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(models.rename(columns={"modelo": "Modelo", "f1_std": "F1 (desv.)", "tiempo_s": "Tiempo (s)"})
                     .round(4), hide_index=True, width="stretch")
    with right:
        source = st.radio("Matriz de confusión", ["Test (modelo final)"] + list(meta["oof_confusions"]),
                          index=0, horizontal=False, label_visibility="visible")
        matrix = meta["test_confusion"] if source.startswith("Test") else meta["oof_confusions"][source]
        fig = px.imshow(matrix, x=["Se queda", "Abandona"], y=["Se queda", "Abandona"], text_auto=True,
                        color_continuous_scale="Blues", labels={"x": "Predicción", "y": "Real", "color": "Clientes"},
                        title=source)
        fig.update_layout(coloraxis_showscale=False)
        show(tidy(fig, 300))

    st.subheader("Estrategias de optimización")
    st.caption("Random Forest, 54 configuraciones por estrategia, mismos folds y métrica F1.")
    optimizers = tables["optimizers"]
    st.dataframe(optimizers.drop(columns="s_por_config").round(4), hide_index=True, width="stretch")
    left, right = st.columns(2)
    long_best = tables["best"].reset_index(names="Configuración").melt("Configuración", var_name="Estrategia", value_name="F1").dropna()
    long_best["Configuración"] += 1
    fig = px.line(long_best, x="Configuración", y="F1", color="Estrategia", title="Mejor F1 acumulado",
                  color_discrete_sequence=[BLUE, ORANGE, AQUA, YELLOW])
    fig.update_traces(line_width=2)
    tidy(fig, 390).update_layout(legend=dict(orientation="h", y=-0.24, x=0, xanchor="left", yanchor="top"))
    with left:
        show(fig)
    fig = px.bar(optimizers.sort_values("tiempo_s"), x="tiempo_s", y="estrategia", orientation="h",
                 title="Tiempo de optimización", labels={"tiempo_s": "Segundos", "estrategia": ""},
                 text=optimizers.sort_values("tiempo_s")["tiempo_s"].map("{:.0f} s".format),
                 color_discrete_sequence=[BLUE])
    fig.update_traces(width=0.55)
    with right:
        show(tidy(fig))

    st.subheader("Frente de Pareto: calidad frente a velocidad")
    pareto = tables["pareto"]
    front = pareto[pareto["no_dominada"]].sort_values("latencia_ms")
    hover = {"n_estimators": True, "max_depth": True, "min_samples_leaf": True, "max_features": True,
             "recall": ":.3f", "precision": ":.3f", "latencia_ms": ":.1f", "f1": ":.4f"}
    fig = px.scatter(pareto[~pareto["no_dominada"]], x="latencia_ms", y="f1", hover_data=hover,
                     labels={"latencia_ms": "Latencia (ms por 1000 clientes)", "f1": "F1 en validación cruzada"},
                     title="Cada punto es una configuración evaluada por NSGA-II")
    fig.update_traces(marker=dict(color=NEUTRAL, size=8), name="Dominadas", showlegend=True)
    fig.add_trace(go.Scatter(x=front["latencia_ms"], y=front["f1"], mode="lines+markers", name="Frente de Pareto",
                             line=dict(color=BLUE, width=2), marker=dict(size=9),
                             customdata=front[["n_estimators", "max_depth", "min_samples_leaf"]],
                             hovertemplate="%{x:.1f} ms, F1 %{y:.4f}<br>árboles %{customdata[0]}, profundidad %{customdata[1]}, "
                                           "hoja %{customdata[2]}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[meta["knee"]["latencia_ms"]], y=[meta["knee"]["f1"]], mode="markers",
                             name="Solución elegida",
                             marker=dict(color=ORANGE, size=16, line=dict(color="white", width=2))))
    show(tidy(fig, 420))


# ------------------------------------------- C. Predicción y explicación
PRESETS = {
    "Cliente en riesgo": dict(tenure=3, MonthlyCharges=85.0, gender="Female", SeniorCitizen="No", Partner="No",
                              Dependents="No", PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic",
                              OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
                              StreamingTV="No", StreamingMovies="No", Contract="Month-to-month",
                              PaperlessBilling="Yes", PaymentMethod="Electronic check"),
    "Cliente fiel": dict(tenure=60, MonthlyCharges=65.0, gender="Male", SeniorCitizen="No", Partner="Yes",
                         Dependents="Yes", PhoneService="Yes", MultipleLines="Yes", InternetService="DSL",
                         OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="No", TechSupport="Yes",
                         StreamingTV="No", StreamingMovies="No", Contract="Two year",
                         PaperlessBilling="No", PaymentMethod="Bank transfer (automatic)"),
}
YES_NO = ["No", "Yes"]
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]


def normalize_services(values):
    """Sin internet o sin teléfono, los servicios dependientes toman la categoría del dataset."""
    out = dict(values)
    if out["InternetService"] == "No":
        out.update({c: "No internet service" for c in INTERNET_ADDONS})
    if out["PhoneService"] == "No":
        out["MultipleLines"] = "No phone service"
    out["TotalCharges"] = out["tenure"] * out["MonthlyCharges"]
    return out


def describe_value(feature, row):
    value = row[feature]
    if feature == "tenure":
        return f"{LABELS[feature]}: {value:.0f} meses"
    if feature in ("MonthlyCharges", "TotalCharges"):
        return f"{LABELS[feature]}: ${value:,.2f}"
    return f"{LABELS[feature]}: {value}"


def contribution_chart(labels, values, title):
    frame = pd.DataFrame({"factor": labels, "peso": values})
    frame = frame.reindex(frame["peso"].abs().sort_values().index)
    fig = go.Figure(go.Bar(x=frame["peso"], y=frame["factor"], orientation="h", width=0.55,
                           marker_color=[ORANGE if v > 0 else BLUE for v in frame["peso"]],
                           hovertemplate="%{y}<br>Efecto: %{x:+.3f}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Naranja sube el riesgo, azul lo baja")
    return tidy(fig, 380)


def section_prediction():
    models, meta = load_models(), load_metadata()
    st.header("Predicción y explicación")
    preset = PRESETS[st.selectbox("Partir de un perfil de ejemplo", list(PRESETS))]

    with st.form("formulario_cliente"):
        st.subheader("Datos del cliente")
        c1, c2, c3 = st.columns(3)
        v = {}
        with c1:
            v["tenure"] = st.slider("Antigüedad (meses)", 0, 72, preset["tenure"])
            v["MonthlyCharges"] = st.number_input("Cargo mensual ($)", 15.0, 125.0, preset["MonthlyCharges"], step=1.0)
            options = ["Month-to-month", "One year", "Two year"]
            v["Contract"] = st.selectbox("Contrato", options, index=options.index(preset["Contract"]))
            v["PaymentMethod"] = st.selectbox("Método de pago", PAYMENTS, index=PAYMENTS.index(preset["PaymentMethod"]))
            v["PaperlessBilling"] = st.radio(LABELS["PaperlessBilling"], YES_NO, index=YES_NO.index(preset["PaperlessBilling"]), horizontal=True)
            options = ["Female", "Male"]
            v["gender"] = st.radio(LABELS["gender"], options, index=options.index(preset["gender"]), horizontal=True)
        with c2:
            options = ["DSL", "Fiber optic", "No"]
            v["InternetService"] = st.selectbox(LABELS["InternetService"], options, index=options.index(preset["InternetService"]))
            for col in INTERNET_ADDONS:
                v[col] = st.radio(LABELS[col], YES_NO, index=YES_NO.index(preset[col]), horizontal=True)
        with c3:
            v["PhoneService"] = st.radio(LABELS["PhoneService"], YES_NO, index=YES_NO.index(preset["PhoneService"]), horizontal=True)
            v["MultipleLines"] = st.radio(LABELS["MultipleLines"], YES_NO, index=YES_NO.index(preset["MultipleLines"]), horizontal=True)
            for col in ["SeniorCitizen", "Partner", "Dependents"]:
                v[col] = st.radio(LABELS[col], YES_NO, index=YES_NO.index(preset[col]), horizontal=True)
            st.caption("El total facturado se estima como antigüedad por cargo mensual.")
        submitted = st.form_submit_button("Calcular riesgo", type="primary")

    if submitted:
        st.session_state["cliente"] = normalize_services(v)
    if "cliente" not in st.session_state:
        st.info("Completa el formulario y pulsa Calcular riesgo.")
        return

    row = pd.DataFrame([st.session_state["cliente"]])[FEATURES]
    raw = float(models["final"].predict_proba(row)[0, 1])
    prob = float(calibrate(models["calibrator"], [raw])[0])
    threshold = meta["threshold_calibrated"]
    band = risk_band(prob, threshold)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilidad de abandono", percent(prob))
    m2.metric("Nivel de riesgo", band)
    m3.metric("Decisión con el umbral del modelo", "Priorizar" if prob >= threshold else "No priorizar")
    st.progress(min(prob, 1.0))

    transformed = models["final"].named_steps["prep"].transform(row)
    shap_values = models["explainer"].shap_values(transformed)[0, :, 1]
    grouped = group_by_original(shap_values, models["names"]).sort_values(key=np.abs, ascending=False)
    record = row.iloc[0]

    st.subheader("Por qué, en palabras simples")
    st.write(f"De cada 100 clientes con un perfil parecido, el modelo estima que unos **{prob * 100:.0f}** se irían.")
    rising = grouped[grouped > 0].head(3)
    falling = grouped[grouped < 0].head(3)
    left, right = st.columns(2)
    with left:
        st.markdown("**Lo que más aumenta su riesgo**")
        st.markdown(literal_dollars("\n".join(f"- {describe_value(f, record)}" for f in rising.index)) or "Ningún factor destacado.")
    with right:
        st.markdown("**Lo que más lo reduce**")
        st.markdown(literal_dollars("\n".join(f"- {describe_value(f, record)}" for f in falling.index)) or "Ningún factor destacado.")
    st.caption("Estos factores describen en qué se fija el modelo. No prueban que cambiarlos evite el abandono.")

    lime_explainer, maps = load_lime()
    encoded = lime_encode(row, maps)[0]
    with st.spinner("Calculando explicación LIME..."):
        explanation = lime_explainer.explain_instance(
            encoded, lambda a: models["final"].predict_proba(lime_decode(a, maps)),
            labels=(1,), num_features=8, num_samples=1500)

    left, right = st.columns(2)
    top = grouped.head(8)
    with left:
        show(contribution_chart([describe_value(f, record) for f in top.index], top.values, "SHAP: aporte de cada variable"))
    with right:
        weights = explanation.as_list(label=1)
        show(contribution_chart([lime_label(t) for t, _ in weights], [w for _, w in weights], "LIME: aproximación local"))
    st.caption("SHAP y LIME explican la salida directa del bosque, antes de calibrar. Por eso sus escalas no suman la "
               "probabilidad mostrada arriba, aunque el sentido de cada factor es el mismo.")

    with st.expander("Comparar con otros modelos exportados"):
        st.caption("Salida directa de cada modelo, entrenado con clases balanceadas y sin calibrar.")
        st.dataframe(pd.DataFrame({
            "Modelo": ["Random Forest final", "Logistic Regression", "CatBoost"],
            "Puntuación": [raw, float(models["logreg"].predict_proba(row)[0, 1]),
                           float(models["catboost"].predict_proba(row)[0, 1])],
        }).round(3), hide_index=True)


# ------------------------------------------------- D. Aplicación de negocio
ACTIONS = {
    "Contract": "Ofrecer migración a plan anual con beneficio",
    "tenure": "Activar acompañamiento de primeros meses",
    "InternetService": "Revisar calidad del servicio y plan contratado",
    "PaymentMethod": "Incentivar el pago automático",
    "TechSupport": "Ofrecer prueba de soporte técnico",
    "OnlineSecurity": "Ofrecer prueba de seguridad en línea",
    "MonthlyCharges": "Revisar si su plan se ajusta a su consumo",
    "TotalCharges": "Reconocer la relación con el cliente",
    "PaperlessBilling": "Revisar su experiencia de facturación",
}
DEFAULT_ACTION = "Llamada de seguimiento"


def score_upload(file, models):
    data = pd.read_csv(file)
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        st.error("Faltan columnas en el CSV: " + ", ".join(missing))
        return None
    data = clean(data)
    numeric_ok = data[["tenure", "MonthlyCharges"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    if not numeric_ok.all():
        st.warning(f"Se omitieron {int((~numeric_ok).sum())} filas con antigüedad o cargo no numéricos.")
    data = data[numeric_ok].reset_index(drop=True)
    if data.empty:
        st.error("El CSV no tiene filas válidas.")
        return None
    data["tenure"] = pd.to_numeric(data["tenure"])
    data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"])
    if ID_COL not in data.columns:
        data[ID_COL] = [f"cliente-{i + 1}" for i in range(len(data))]
    raw = models["final"].predict_proba(data[FEATURES])[:, 1]
    data["p_churn"] = calibrate(models["calibrator"], raw)
    return data


def main_reasons(rows, models):
    """Motivo principal por cliente: la variable con mayor aporte positivo en SHAP."""
    if rows.empty:
        return []
    transformed = models["final"].named_steps["prep"].transform(rows[FEATURES])
    values = models["explainer"].shap_values(transformed)[:, :, 1]
    columns = [original_feature(name) for name in models["names"]]
    return pd.DataFrame(values, columns=columns).T.groupby(level=0).sum().T.idxmax(axis=1).tolist()


def section_business():
    models, tables, meta = load_models(), load_tables(), load_metadata()
    st.header("Aplicación de negocio")
    st.caption("Ancla ordena la cartera de un ISP y reparte un presupuesto limitado entre los clientes donde más vale actuar.")

    source = st.radio("Cartera", ["Demostración (clientes de test)", "Subir CSV propio"], horizontal=True)
    portfolio = tables["scored"]
    has_labels = True
    if source.startswith("Subir"):
        template = tables["scored"].head(5)[[ID_COL] + FEATURES].to_csv(index=False).encode("utf-8")
        st.download_button("Descargar plantilla CSV", template, "plantilla_cartera.csv", "text/csv")
        file = st.file_uploader("CSV con las columnas de la plantilla", type="csv")
        if file is None:
            st.info("Sube un archivo para puntuar tu cartera.")
            return
        try:
            portfolio = score_upload(file, models)
        except Exception as error:
            st.error(f"No se pudo leer el archivo: {error}")
            return
        if portfolio is None:
            return
        has_labels = False

    st.subheader("Supuestos de la campaña")
    st.caption("**Valores hipotéticos** que cada ISP debe sustituir por los suyos. No provienen del modelo ni del dataset.")
    c1, c2, c3 = st.columns(3)
    max_share = c1.slider("Cartera máxima a intervenir (%)", 1, 100, 10) / 100
    budget = c1.number_input("Presupuesto de ofertas ($)", 0, 200000, 2500, step=100)
    offer_cost = c2.number_input("Coste por oferta ($)", 1, 500, 15)
    success = c2.slider("Tasa de éxito de la oferta (%)", 1, 80, 20) / 100
    horizon = c3.slider("Horizonte de ingreso (meses)", 1, 24, 12)
    subscription = c3.number_input("Suscripción de Ancla ($/mes)", 0, 5000, 199, step=10)

    plan = plan_campaign(portfolio, budget, offer_cost, success, horizon, max_share)
    software = subscription * horizon
    summary = campaign_summary(plan, offer_cost, software)

    st.subheader("Resultado estimado con esos supuestos")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clientes a contactar", f"{summary['contactados']:,} de {len(plan):,}")
    k2.metric("Ingreso salvado esperado", money(summary["ingreso_salvado"]))
    k3.metric("Beneficio neto", money(summary["beneficio_neto"]))
    k4.metric("ROI", f"{summary['roi']:.0%}")
    st.caption(literal_dollars(f"Estimación hipotética a {horizon} meses. Costes: ofertas {money(summary['gasto_ofertas'])} "
                               f"y software {money(software)}."))

    chosen = plan[plan["contactar"]]
    if has_labels and len(chosen):
        st.subheader("Evidencia real del modelo en esta cartera")
        real = chosen["churn_real"].mean()
        base = plan["churn_real"].mean()
        e1, e2, e3 = st.columns(3)
        e1.metric("Abandonan entre los contactados", percent(real))
        e2.metric("Abandonan si se elige al azar", percent(base))
        e3.metric("Abandonos capturados", percent(chosen["churn_real"].sum() / plan["churn_real"].sum()))
        st.caption("Resultado medido con clientes de test que el modelo no vio. No depende de los supuestos económicos, "
                   "salvo por la cantidad de clientes elegidos.")

    left, right = st.columns(2)
    curve = plan.assign(acumulado=plan["valor_neto_esperado"].cumsum() - software)
    curve["cartera"] = (curve.index + 1) / len(curve)
    fig = px.line(curve, x="cartera", y="acumulado", title="Beneficio neto según la cartera contactada (hipotético)",
                  labels={"cartera": "Cartera contactada, de mayor a menor valor", "acumulado": "Beneficio neto ($)"},
                  color_discrete_sequence=[BLUE])
    fig.update_traces(line_width=2)
    fig.update_xaxes(tickformat=".0%")
    if len(chosen):
        fig.add_trace(go.Scatter(x=[len(chosen) / len(plan)], y=[curve["acumulado"].iloc[len(chosen) - 1]], mode="markers",
                                 name="Plan actual", marker=dict(color=ORANGE, size=13, line=dict(color="white", width=2))))
    with left:
        show(tidy(fig))

    scenarios = []
    for name, rate in [("Pesimista", success * 0.5), ("Base", success), ("Optimista", min(success * 1.5, 1.0))]:
        outcome = campaign_summary(plan_campaign(portfolio, budget, offer_cost, rate, horizon, max_share), offer_cost, software)
        scenarios.append({"Escenario": name, "Tasa de éxito": f"{rate:.0%}", "Contactados": outcome["contactados"],
                          "Ingreso salvado": money(outcome["ingreso_salvado"]),
                          "Beneficio neto": money(outcome["beneficio_neto"]), "ROI": f"{outcome['roi']:.0%}"})
    with right:
        st.markdown("**Escenarios hipotéticos**")
        st.dataframe(pd.DataFrame(scenarios), hide_index=True, width="stretch")
        st.caption("La tasa de éxito es el supuesto más sensible. Solo un piloto con grupo de control puede medirla.")

    st.subheader("Lista priorizada")
    if chosen.empty:
        st.warning("Con estos supuestos ningún contacto se paga solo. Sube el presupuesto o revisa el coste de la oferta.")
        return
    full = chosen.copy()
    reasons = main_reasons(full, models)
    full["Motivo principal"] = [LABELS[r] for r in reasons]
    full["Acción a evaluar"] = [ACTIONS.get(r, DEFAULT_ACTION) for r in reasons]
    full["Riesgo"] = [risk_band(p, meta["threshold_calibrated"]) for p in full["p_churn"]]
    output = full[[ID_COL, "Riesgo", "p_churn", "MonthlyCharges", "ingreso_en_riesgo", "valor_neto_esperado",
                      "Motivo principal", "Acción a evaluar"]].rename(columns={
                          ID_COL: "Cliente", "p_churn": "Prob. abandono", "MonthlyCharges": "Cargo mensual",
                          "ingreso_en_riesgo": "Ingreso en riesgo", "valor_neto_esperado": "Valor neto esperado"})
    st.dataframe(output.head(300).round(2), hide_index=True, width="stretch", height=360)
    note = " Se muestran los primeros 300; la descarga incluye todos." if len(chosen) > 300 else ""
    st.caption("Las acciones son hipótesis a probar: el modelo indica asociación, no causa." + note)
    st.download_button("Descargar lista priorizada", output.to_csv(index=False).encode("utf-8"), "lista_priorizada.csv", "text/csv")


def main():
    st.sidebar.title("Ancla")
    st.sidebar.caption("Retención de clientes para ISP regionales")
    requested = st.query_params.get("seccion", "1")
    start = int(requested) - 1 if requested in {"1", "2", "3", "4"} else 0
    section = st.sidebar.radio("Sección", SECTIONS, index=start)
    st.sidebar.divider()
    st.sidebar.caption("Modelo entrenado con un dataset público de una telco ficticia de EE. UU. "
                       "Para usarlo en un ISP real hay que reentrenarlo con sus datos.")
    {
        SECTIONS[0]: section_eda,
        SECTIONS[1]: section_performance,
        SECTIONS[2]: section_prediction,
        SECTIONS[3]: section_business,
    }[section]()


main()
