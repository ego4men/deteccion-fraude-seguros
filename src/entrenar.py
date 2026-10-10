"""
Entrena y evalúa modelos de detección de fraude.

Decisiones clave (y por qué):
    * Separación TEMPORAL: entrena con 1994-1995 y prueba con 1996. Un modelo de fraude
      siempre se usa sobre reclamos futuros; mezclar años sobreestimaría su desempeño.
    * Métrica principal PR-AUC (precisión-sensibilidad), no exactitud: con 6 % de fraude,
      un modelo que dice "nunca hay fraude" tendría 94 % de exactitud y no serviría de nada.
    * El umbral de decisión se elige con predicciones "fuera de pliegue" del ENTRENAMIENTO
      (validación cruzada), nunca mirando el conjunto de prueba.
    * CALIBRACIÓN: scale_pos_weight compensa el desbalance pero infla las probabilidades
      (un reclamo típico saldría con 50 %+). El escalado de Platt las devuelve a valores
      reales sin cambiar el orden de los reclamos, así que el ranking no se altera.

Uso:
    python src/entrenar.py
"""
import json

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from datos import BINARIAS, NOMINALES, NUMERICAS, ORDINALES, RAIZ, cargar, separar_por_anio
from eda import AZUL, GRIS, NARANJA, TEXTO, TEXTO_2  # mismo estilo de gráficos

SEMILLA = 42
IMG = RAIZ / "img"
MODELOS = RAIZ / "modelos"
RESULTADOS = RAIZ / "resultados"
for carpeta in (IMG, MODELOS, RESULTADOS):
    carpeta.mkdir(exist_ok=True)

# Supuesto de negocio (editable también en la app): investigar un reclamo cuesta
# 1 unidad; dejar pasar un fraude cuesta 20 unidades (pago indebido promedio).
COSTO_INVESTIGAR = 1
COSTO_FRAUDE_NO_DETECTADO = 20


def captura_en_top(y_true, prob, fraccion):
    """% del fraude total que se encuentra revisando solo el top `fraccion` de reclamos."""
    n = int(np.ceil(len(prob) * fraccion))
    indices = np.argsort(-prob)[:n]
    return np.asarray(y_true)[indices].sum() / np.asarray(y_true).sum()


def umbral_costo_minimo(y_true, prob):
    """Umbral que minimiza: (reclamos investigados × costo) + (fraudes no detectados × costo)."""
    y_true = np.asarray(y_true)
    mejores = []
    for u in np.linspace(0.005, 0.5, 100):
        marcados = prob >= u
        costo = marcados.sum() * COSTO_INVESTIGAR + (y_true[~marcados] == 1).sum() * COSTO_FRAUDE_NO_DETECTADO
        mejores.append((costo, u))
    return min(mejores)[1]


def main():
    df = cargar()
    X_ent, X_prueba, y_ent, y_prueba = separar_por_anio(df)
    print(f"Entrenamiento: {len(y_ent):,} reclamos ({y_ent.mean():.2%} fraude) · "
          f"Prueba 1996: {len(y_prueba):,} ({y_prueba.mean():.2%} fraude)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    peso = (y_ent == 0).sum() / (y_ent == 1).sum()   # compensa el desbalance de clases

    # 1) Línea base: regresión logística
    logistica = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5),
    )
    logistica.fit(X_ent, y_ent)

    # 2) XGBoost con búsqueda de hiperparámetros en validación cruzada
    busqueda = GridSearchCV(
        XGBClassifier(n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=peso, eval_metric="aucpr", random_state=SEMILLA, n_jobs=-1),
        param_grid={"max_depth": [3, 4, 6], "min_child_weight": [1, 5]},
        scoring="average_precision", cv=cv, n_jobs=1,
    )
    busqueda.fit(X_ent, y_ent)
    xgb_base = busqueda.best_estimator_
    print("Mejores hiperparámetros:", busqueda.best_params_,
          f"· PR-AUC en validación cruzada: {busqueda.best_score_:.3f}")
    xgb = CalibratedClassifierCV(xgb_base, method="sigmoid", cv=cv)
    xgb.fit(X_ent, y_ent)

    # 3) Umbral elegido con predicciones fuera de pliegue (solo entrenamiento)
    prob_oof = cross_val_predict(xgb, X_ent, y_ent, cv=cv, method="predict_proba")[:, 1]
    umbral = umbral_costo_minimo(y_ent, prob_oof)

    # 4) Evaluación final en 1996 (datos nunca vistos)
    resultados = {}
    probs = {}
    for nombre, modelo in [("Regresión logística", logistica), ("XGBoost", xgb)]:
        p = modelo.predict_proba(X_prueba)[:, 1]
        probs[nombre] = p
        resultados[nombre] = {
            "roc_auc": roc_auc_score(y_prueba, p),
            "pr_auc": average_precision_score(y_prueba, p),
            "captura_top10": captura_en_top(y_prueba, p, 0.10),
            "captura_top20": captura_en_top(y_prueba, p, 0.20),
            "captura_top30": captura_en_top(y_prueba, p, 0.30),
        }
    p = probs["XGBoost"]
    pred = (p >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_prueba, pred).ravel()
    costo_modelo = (tp + fp) * COSTO_INVESTIGAR + fn * COSTO_FRAUDE_NO_DETECTADO
    costo_sin_modelo = y_prueba.sum() * COSTO_FRAUDE_NO_DETECTADO          # no investigar nada
    costo_todo = len(y_prueba) * COSTO_INVESTIGAR                            # investigar todo
    resultados["XGBoost"].update({
        "umbral": umbral, "precision": precision_score(y_prueba, pred), "recall": recall_score(y_prueba, pred),
        "vp": int(tp), "fp": int(fp), "fn": int(fn), "vn": int(tn),
        "pct_investigado": float(pred.mean()),
        "costo_modelo": float(costo_modelo), "costo_sin_modelo": float(costo_sin_modelo),
        "costo_investigar_todo": float(costo_todo),
        "ahorro_vs_mejor_alternativa": float(1 - costo_modelo / min(costo_sin_modelo, costo_todo)),
    })
    tabla = pd.DataFrame(resultados).T
    print(tabla.to_string(float_format=lambda x: f"{x:.3f}"))

    # 5) Guardar modelo y metadatos para la app
    joblib.dump(xgb, MODELOS / "xgb_fraude.joblib")
    importancia = pd.Series(xgb_base.feature_importances_, index=X_ent.columns).sort_values(ascending=False)
    meta = {
        "columnas": list(X_ent.columns), "umbral": float(umbral),
        "costo_investigar": COSTO_INVESTIGAR, "costo_fraude_no_detectado": COSTO_FRAUDE_NO_DETECTADO,
        "hiperparametros": busqueda.best_params_, "metricas_prueba_1996": resultados,
        "tasa_fraude_prueba": float(y_prueba.mean()),
    }
    (MODELOS / "metadatos.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    # Opciones y valor más frecuente de cada variable original: la app los usa
    # para armar el formulario sin necesitar el dataset crudo.
    columnas_crudas = NUMERICAS + list(ORDINALES) + list(BINARIAS) + NOMINALES + ["Month", "MonthClaimed"]
    opciones = {
        col: {"opciones": sorted(df[col].dropna().unique().tolist(), key=str),
              "frecuente": df[col].mode().iloc[0]}
        for col in columnas_crudas
    }
    (MODELOS / "opciones_formulario.json").write_text(
        json.dumps(opciones, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    importancia.to_csv(RESULTADOS / "importancia_variables.csv", header=["importancia"])
    pd.DataFrame({"y": y_prueba.values, "prob": p}).to_csv(RESULTADOS / "predicciones_1996.csv", index=False)

    # 6) Gráficos
    # 6a. Curvas precisión-sensibilidad
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for (nombre, pr), color in zip(probs.items(), [GRIS, AZUL]):
        prec, rec, _ = precision_recall_curve(y_prueba, pr)
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{nombre} (PR-AUC {resultados[nombre]['pr_auc']:.3f})")
    ax.axhline(y_prueba.mean(), color=TEXTO_2, linestyle="--", linewidth=1)
    ax.text(0.55, y_prueba.mean() - 0.04, f"Azar ({y_prueba.mean():.1%})", ha="center", color=TEXTO_2, fontsize=9)
    ax.set_xlabel("Sensibilidad (recall): % del fraude detectado")
    ax.set_ylabel("Precisión: % de alertas que son fraude")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Curva precisión-sensibilidad · prueba 1996")
    fig.savefig(IMG / "03_curva_pr.png")
    plt.close(fig)

    # 6b. Curva de captura (ganancia acumulada)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    fracciones = np.linspace(0, 1, 101)
    for (nombre, pr), color in zip(probs.items(), [GRIS, AZUL]):
        ax.plot(fracciones, [captura_en_top(y_prueba, pr, f) if f > 0 else 0 for f in fracciones],
                color=color, linewidth=2, label=nombre)
    ax.plot([0, 1], [0, 1], color=TEXTO_2, linestyle="--", linewidth=1, label="Revisión al azar")
    c20 = resultados["XGBoost"]["captura_top20"]
    ax.scatter([0.2], [c20], color=NARANJA, zorder=3, s=40)
    ax.annotate(f"Revisando el 20 % de reclamos\nse detecta el {c20:.0%} del fraude", (0.2, c20),
                xytext=(0.32, c20 - 0.18), color=TEXTO, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": TEXTO_2})
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_xlabel("% de reclamos revisados (ordenados por riesgo)")
    ax.set_ylabel("% del fraude detectado")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Curva de captura · prueba 1996")
    fig.savefig(IMG / "04_curva_captura.png")
    plt.close(fig)

    # 6c. Importancia de variables
    top = importancia.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(top.index, top.values, color=AZUL, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Importancia (ganancia promedio en XGBoost)")
    ax.set_title("Variables más importantes del modelo")
    fig.savefig(IMG / "05_importancia_variables.png")
    plt.close(fig)

    # 6d. Matriz de confusión con el umbral de costo mínimo
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    matriz = np.array([[tn, fp], [fn, tp]])
    ax.imshow(matriz, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matriz[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if matriz[i, j] > matriz.max() / 2 else TEXTO)
    ax.set_xticks([0, 1], ["No alerta", "Alerta"])
    ax.set_yticks([0, 1], ["Legítimo", "Fraude"])
    ax.grid(False)
    ax.set_title(f"Matriz de confusión (umbral {umbral:.1%})")
    fig.savefig(IMG / "06_matriz_confusion.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
