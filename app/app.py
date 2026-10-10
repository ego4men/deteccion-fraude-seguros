"""
App de detección de fraude en reclamos de seguro vehicular.

Ejecutar desde la carpeta del proyecto:
    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from datos import preparar_variables  # noqa: E402

st.set_page_config(page_title="Detección de fraude en seguros", page_icon="🛡️", layout="wide")


@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(RAIZ / "modelos" / "xgb_fraude.joblib")
    meta = json.loads((RAIZ / "modelos" / "metadatos.json").read_text(encoding="utf-8"))
    opciones = json.loads((RAIZ / "modelos" / "opciones_formulario.json").read_text(encoding="utf-8"))
    predicciones = pd.read_csv(RAIZ / "resultados" / "predicciones_1996.csv")
    return modelo, meta, opciones, predicciones


modelo, meta, opciones, predicciones = cargar_modelo()
xgb = meta["metricas_prueba_1996"]["XGBoost"]

st.title("🛡️ Detección de fraude en reclamos de seguro vehicular")
st.caption("Modelo XGBoost entrenado con reclamos de 1994-1995 y evaluado con 1996 · "
           "Dataset público *Vehicle Insurance Claim Fraud Detection* (Kaggle)")

pestana_reclamo, pestana_costos, pestana_modelo = st.tabs(
    ["🔎 Evaluar un reclamo", "💰 Simulador de costos", "📊 Sobre el modelo"])

# ---------------------------------------------------------------------------
# 1. Evaluar un reclamo
# ---------------------------------------------------------------------------
ETIQUETAS = {
    "Fault": "¿Quién fue responsable del accidente?",
    "BasePolicy": "Cobertura",
    "VehicleCategory": "Categoría del vehículo",
    "PolicyType": "Tipo de póliza",
    "AddressChange_Claim": "Cambio de domicilio antes del reclamo",
    "VehiclePrice": "Precio del vehículo (USD)",
    "Deductible": "Deducible (USD)",
    "PastNumberOfClaims": "Reclamos anteriores",
    "AgeOfPolicyHolder": "Edad del asegurado",
    "Make": "Marca",
    "AgentType": "Tipo de agente",
    "PoliceReportFiled": "¿Hubo denuncia policial?",
}


def selector(col, contenedor):
    lista = opciones[col]["opciones"]
    frecuente = opciones[col]["frecuente"]
    return contenedor.selectbox(ETIQUETAS[col], lista, index=lista.index(frecuente), key=col)


with pestana_reclamo:
    st.write("Completa los datos principales del reclamo. Los demás campos toman su valor más frecuente.")
    c1, c2, c3 = st.columns(3)
    valores = {}
    for i, col in enumerate(ETIQUETAS):
        valores[col] = selector(col, [c1, c2, c3][i % 3])

    fila = {col: datos["frecuente"] for col, datos in opciones.items()}
    fila.update(valores)
    X = preparar_variables(pd.DataFrame([fila])).reindex(columns=meta["columnas"], fill_value=0)
    prob = float(modelo.predict_proba(X)[:, 1][0])
    umbral = meta["umbral"]

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilidad de fraude estimada", f"{prob:.1%}")
    m2.metric("Umbral de alerta", f"{umbral:.1%}")
    riesgo_relativo = prob / meta["tasa_fraude_prueba"]
    m3.metric("Riesgo frente al promedio", f"{riesgo_relativo:.1f} veces")
    if prob >= umbral:
        st.error("🚩 **Enviar a investigación**: la probabilidad supera el umbral de alerta.")
    else:
        st.success("✅ **Aprobar por la vía regular**: la probabilidad está por debajo del umbral.")
    st.caption("La probabilidad es una estimación estadística, no una acusación: la decisión final "
               "corresponde al analista de siniestros.")

# ---------------------------------------------------------------------------
# 2. Simulador de costos: ¿qué umbral conviene según el costo de cada error?
# ---------------------------------------------------------------------------
with pestana_costos:
    st.write("Elige cuánto cuesta **investigar** un reclamo y cuánto se pierde si un **fraude pasa sin "
             "detectarse**. La app busca el umbral que minimiza el costo total en los reclamos de 1996.")
    s1, s2 = st.columns(2)
    costo_inv = s1.number_input("Costo de investigar un reclamo (USD)", 10, 5000, 150, step=10)
    costo_fraude = s2.number_input("Pérdida promedio por fraude no detectado (USD)", 100, 100000, 3000, step=100)

    y, p = predicciones["y"].to_numpy(), predicciones["prob"].to_numpy()
    umbrales = np.linspace(0.005, 0.5, 100)
    costos = [((p >= u).sum() * costo_inv + ((p < u) & (y == 1)).sum() * costo_fraude) for u in umbrales]
    mejor = umbrales[int(np.argmin(costos))]
    alerta = p >= mejor
    costo_min = min(costos)
    alternativa = min(len(y) * costo_inv, y.sum() * costo_fraude)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Umbral óptimo", f"{mejor:.1%}")
    k2.metric("Reclamos a investigar", f"{alerta.mean():.1%}")
    k3.metric("Fraude detectado", f"{(alerta & (y == 1)).sum() / y.sum():.1%}")
    k4.metric("Ahorro vs. mejor alternativa sin modelo", f"{1 - costo_min / alternativa:.1%}")
    st.line_chart(pd.DataFrame({"Umbral": umbrales, "Costo total (USD)": costos}).set_index("Umbral"))
    st.caption("Mejor alternativa sin modelo = la más barata entre investigar todos los reclamos o no investigar ninguno.")

# ---------------------------------------------------------------------------
# 3. Sobre el modelo
# ---------------------------------------------------------------------------
with pestana_modelo:
    a, b, c, d = st.columns(4)
    a.metric("ROC-AUC (1996)", f"{xgb['roc_auc']:.3f}")
    b.metric("PR-AUC (1996)", f"{xgb['pr_auc']:.3f}", f"{xgb['pr_auc'] / meta['tasa_fraude_prueba']:.1f}x el azar")
    c.metric("Fraude detectado revisando el 20 %", f"{xgb['captura_top20']:.0%}")
    d.metric("Recall con umbral de costo mínimo", f"{xgb['recall']:.0%}")
    g1, g2 = st.columns(2)
    g1.image(str(RAIZ / "img" / "04_curva_captura.png"))
    g2.image(str(RAIZ / "img" / "05_importancia_variables.png"))
    st.markdown("Código y metodología completos en el "
                "[repositorio del proyecto](https://github.com/ego4men/deteccion-fraude-seguros).")
