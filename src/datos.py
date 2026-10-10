"""
Descarga, limpieza y preparación de variables del dataset de fraude en seguros vehiculares.

Fuente original: Kaggle · "Vehicle Insurance Claim Fraud Detection" (shivamb), archivo fraud_oracle.csv.
El CSV no se versiona en el repositorio: se descarga y se verifica con su hash SHA-256.

Uso:
    python src/datos.py          # descarga data/fraud_oracle.csv si no existe
"""
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CSV = RAIZ / "data" / "fraud_oracle.csv"
URL_COPIA = ("https://media.githubusercontent.com/media/lulucopter/"
             "dsa4263-fraud-detection/HEAD/data/raw/fraud_oracle.csv")
SHA256 = "8b6aa59764ef4f8b058598d3e8f325ef3623f52f4b1cd4ef946baebb5ccfa9a6"
OBJETIVO = "FraudFound_P"


def descargar() -> None:
    """Descarga el CSV y comprueba que sea exactamente el archivo esperado."""
    if RUTA_CSV.exists():
        return
    RUTA_CSV.parent.mkdir(exist_ok=True)
    contenido = urllib.request.urlopen(URL_COPIA, timeout=60).read()
    if hashlib.sha256(contenido).hexdigest() != SHA256:
        raise ValueError("El archivo descargado no coincide con el original (hash distinto).")
    RUTA_CSV.write_bytes(contenido)


def cargar() -> pd.DataFrame:
    """Lee el CSV y corrige los problemas de calidad detectados en la exploración."""
    descargar()
    df = pd.read_csv(RUTA_CSV, encoding="utf-8-sig")
    # 1 fila con día/mes de reclamo = "0" (registro inválido)
    df = df[df.DayOfWeekClaimed != "0"].copy()
    # 320 filas con Age = 0: la edad exacta falta, pero el rango AgeOfPolicyHolder sí existe.
    # Se marca como faltante para no enseñarle al modelo que existen asegurados de 0 años.
    df["Age"] = df["Age"].replace(0, pd.NA).astype("Float64")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Variables: los rangos de texto ("1 to 7", "more than 30") se convierten a
# números ordenados, porque su orden tiene significado.
# ---------------------------------------------------------------------------
ORDINALES = {
    "VehiclePrice": {"less than 20000": 0, "20000 to 29000": 1, "30000 to 39000": 2,
                     "40000 to 59000": 3, "60000 to 69000": 4, "more than 69000": 5},
    "Days_Policy_Accident": {"none": 0, "1 to 7": 1, "8 to 15": 2, "15 to 30": 3, "more than 30": 4},
    "Days_Policy_Claim": {"none": 0, "8 to 15": 1, "15 to 30": 2, "more than 30": 3},
    "PastNumberOfClaims": {"none": 0, "1": 1, "2 to 4": 2, "more than 4": 3},
    "AgeOfVehicle": {"new": 0, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5,
                     "6 years": 6, "7 years": 7, "more than 7": 8},
    "AgeOfPolicyHolder": {"16 to 17": 0, "18 to 20": 1, "21 to 25": 2, "26 to 30": 3, "31 to 35": 4,
                          "36 to 40": 5, "41 to 50": 6, "51 to 65": 7, "over 65": 8},
    "NumberOfSuppliments": {"none": 0, "1 to 2": 1, "3 to 5": 2, "more than 5": 3},
    "AddressChange_Claim": {"no change": 0, "4 to 8 years": 1, "2 to 3 years": 2,
                            "1 year": 3, "under 6 months": 4},
    "NumberOfCars": {"1 vehicle": 0, "2 vehicles": 1, "3 to 4": 2, "5 to 8": 3, "more than 8": 4},
}
BINARIAS = {
    "AccidentArea": "Urban", "Sex": "Male", "Fault": "Policy Holder",
    "PoliceReportFiled": "Yes", "WitnessPresent": "Yes", "AgentType": "External",
}
NOMINALES = ["Make", "MaritalStatus", "PolicyType", "VehicleCategory", "BasePolicy"]
NUMERICAS = ["Age", "Deductible", "DriverRating", "WeekOfMonth", "WeekOfMonthClaimed"]
# Se descartan: PolicyNumber (identificador), RepNumber (código interno del
# tramitador), Year (se usa para separar entrenamiento/prueba) y fechas sueltas.


def preparar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte el dataset limpio en una matriz numérica lista para el modelo."""
    X = pd.DataFrame(index=df.index)
    for col in NUMERICAS:
        X[col] = df[col].astype("float64")
    for col, mapa in ORDINALES.items():
        X[col] = df[col].map(mapa).astype("float64")
    for col, positivo in BINARIAS.items():
        X[f"{col}_{positivo.replace(' ', '')}"] = (df[col] == positivo).astype(int)
    # Variable derivada: ¿el reclamo se hizo en un mes distinto al del accidente?
    X["ReclamoOtroMes"] = (df["Month"] != df["MonthClaimed"]).astype(int)
    X = X.join(pd.get_dummies(df[NOMINALES], prefix=NOMINALES, dtype=int))
    return X


def separar_por_anio(df: pd.DataFrame):
    """Entrena con 1994-1995 y evalúa con 1996: simula predecir reclamos futuros."""
    X = preparar_variables(df)
    y = df[OBJETIVO]
    entrenamiento = df["Year"] < 1996
    return X[entrenamiento], X[~entrenamiento], y[entrenamiento], y[~entrenamiento]


if __name__ == "__main__":
    datos = cargar()
    print(f"Filas: {len(datos):,} · Fraudes: {datos[OBJETIVO].sum():,} ({datos[OBJETIVO].mean():.2%})")
