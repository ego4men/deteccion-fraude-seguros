"""
Análisis exploratorio (EDA) con pruebas estadísticas.

Pregunta: ¿qué características de la póliza y del siniestro se asocian con el fraude?
    * Prueba chi-cuadrado de independencia: ¿la tasa de fraude cambia según la categoría
      más de lo que explicaría el azar? (p-valor < 0,05 → asociación significativa)
    * V de Cramér: QUÉ TAN fuerte es esa asociación (0 = nada, 1 = total). Con miles de
      filas casi todo sale "significativo"; la V permite ordenar por importancia práctica.

Uso:
    python src/eda.py      → imprime la tabla y guarda gráficos en img/
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from datos import OBJETIVO, RAIZ, cargar

IMG = RAIZ / "img"
IMG.mkdir(exist_ok=True)
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#898781"
TEXTO, TEXTO_2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 10, "axes.edgecolor": GRILLA,
    "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRILLA, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": TEXTO,
    "axes.titlelocation": "left", "figure.dpi": 150, "savefig.bbox": "tight",
})

NOMBRES = {
    "Fault": "Responsable del accidente", "BasePolicy": "Cobertura", "PolicyType": "Tipo de póliza",
    "VehicleCategory": "Categoría del vehículo", "AddressChange_Claim": "Cambio de domicilio",
    "VehiclePrice": "Precio del vehículo", "PastNumberOfClaims": "Reclamos anteriores",
    "Deductible": "Deducible", "AgeOfVehicle": "Antigüedad del vehículo", "Make": "Marca",
    "AgeOfPolicyHolder": "Edad del asegurado", "Sex": "Sexo", "MaritalStatus": "Estado civil",
    "AccidentArea": "Zona del accidente", "AgentType": "Tipo de agente", "NumberOfSuppliments": "Suplementos",
    "PoliceReportFiled": "Denuncia policial", "WitnessPresent": "Testigo presente",
    "Days_Policy_Accident": "Días de póliza al accidente", "Days_Policy_Claim": "Días de póliza al reclamo",
    "NumberOfCars": "N.º de vehículos", "DriverRating": "Calificación del conductor",
    "Month": "Mes del accidente", "DayOfWeek": "Día del accidente",
}


def cramer_v(tabla: pd.DataFrame) -> tuple[float, float]:
    chi2, p, _, _ = chi2_contingency(tabla)
    n = tabla.to_numpy().sum()
    k = min(tabla.shape) - 1
    return float(np.sqrt(chi2 / (n * k))), float(p)


def main():
    df = cargar()
    tasa_global = df[OBJETIVO].mean()

    # 1. Asociación de cada variable categórica con el fraude
    filas = []
    for col in NOMBRES:
        v, p = cramer_v(pd.crosstab(df[col], df[OBJETIVO]))
        filas.append({"variable": col, "nombre": NOMBRES[col], "v_cramer": v, "p_valor": p})
    asociacion = pd.DataFrame(filas).sort_values("v_cramer", ascending=False)
    (RAIZ / "resultados").mkdir(exist_ok=True)
    asociacion.to_csv(RAIZ / "resultados" / "asociacion_fraude.csv", index=False)
    print(asociacion.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Gráfico: top 12 por V de Cramér (gris = no significativa)
    top = asociacion.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colores = [AZUL if p < 0.05 else GRIS for p in top.p_valor]
    ax.barh(top.nombre, top.v_cramer, color=colores, height=0.6)
    for y, v in enumerate(top.v_cramer):
        ax.text(v + 0.003, y, f"{v:.3f}", va="center", color=TEXTO, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("V de Cramér (fuerza de la asociación con el fraude)")
    ax.set_title("Variables más asociadas al fraude (gris = p ≥ 0,05)")
    fig.savefig(IMG / "01_asociacion_fraude.png")
    plt.close(fig)

    # 2. Tasa de fraude en las variables más fuertes
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, col, orden in [
        (axes[0], "Fault", ["Policy Holder", "Third Party"]),
        (axes[1], "BasePolicy", ["All Perils", "Collision", "Liability"]),
        (axes[2], "VehicleCategory", ["Utility", "Sedan", "Sport"]),
    ]:
        tasas = df.groupby(col)[OBJETIVO].mean().reindex(orden)
        ax.bar(tasas.index, tasas.values, color=AZUL, width=0.6)
        ax.axhline(tasa_global, color=TEXTO_2, linestyle="--", linewidth=1)
        for x, v in enumerate(tasas.values):
            ax.text(x, v + 0.002, f"{v:.1%}", ha="center", color=TEXTO, fontsize=9)
        ax.set_title(NOMBRES[col], fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    axes[0].text(1, tasa_global + 0.003, f"Promedio {tasa_global:.1%}", color=TEXTO_2, fontsize=9, ha="center")
    fig.suptitle("Tasa de fraude por categoría", x=0.01, ha="left", fontweight="bold", color=TEXTO)
    fig.tight_layout()
    fig.savefig(IMG / "02_tasa_fraude_categorias.png")
    plt.close(fig)

    # 3. Resumen para el README
    print(f"\nTasa global de fraude: {tasa_global:.2%}")
    for col in ["Fault", "BasePolicy", "AddressChange_Claim", "PastNumberOfClaims", "AgentType"]:
        print(df.groupby(col)[OBJETIVO].agg(["count", "mean"]).round(4), "\n")


if __name__ == "__main__":
    main()
