"""Tarea 3 - Opcion A: DOE factorial 3x3 para ajustar T0 del recocido simulado (TSP)."""
import random
import numpy as np
import pandas as pd
import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tsp_utils import *          

# ---------------- Configuración ----------------
MATRICULA = 2101295         
T_MIN, T_MAX = 1.0, 100.0        # rango elegido (distancia promedio entre ciudades ~52-67, ver escala.py)
TAMANOS = [5, 10, 15]
REPLICAS = 10
ITERS = 200                      # presupuesto chico a propósito: con muchas iteraciones SA siempre llega al óptimo y T0 no se nota
SCHEDULE = "exponencial"       
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results") + os.sep
os.makedirs(OUT, exist_ok=True)

# ---------------- 1. r fijo y niveles de temperatura ----------------
random.seed(MATRICULA)
r = random.random()
fracciones = {"bajo": 0.3 * r, "medio": 0.3 * r + 0.3, "alto": 0.3 * r + 0.6}
temps = {k: round(T_MIN + f * (T_MAX - T_MIN), 2) for k, f in fracciones.items()}
print(f"r = {r:.4f}")
print("Niveles de temperatura:", temps)

# ---------------- 2. Una instancia por tamaño (misma que escala.py) ----------------
instancias = {}
for n in TAMANOS:
    rng = np.random.default_rng(7 + n)
    instancias[n] = dist_matrix(random_cities(n, rng))

# ---------------- 3. Experimento: 3 x 3 x 10 = 90 corridas ----------------
filas = []
for n in TAMANOS:
    for nivel, T0 in temps.items():
        for rep in range(REPLICAS):
            seed = MATRICULA + 1000 * n + 100 * list(temps).index(nivel) + rep   # semilla distinta por corrida
            _, costo, _ = simulated_annealing(instancias[n], SCHEDULE, iters=ITERS, T0=T0, seed=seed)
            filas.append({"nivel": nivel, "temperatura": T0, "tamano": n, "costo": costo})
df = pd.DataFrame(filas)
df.to_csv(OUT + "resultados_doe.csv", index=False)
print(df.groupby(["tamano", "temperatura"])["costo"].agg(["mean", "std", "min"]).round(2))

# ---------------- 4. ANOVA de dos factores ----------------
modelo = ols("costo ~ C(temperatura) * C(tamano)", data=df).fit()
tabla_anova = sm.stats.anova_lm(modelo, typ=2)
print("\nANOVA dos factores (alpha = 0.05):\n", tabla_anova)
tabla_anova.to_csv(OUT + "anova_doe.csv")

# ---------------- 5. Bootstrap de la mediana (IC 95%) ----------------
def boot_mediana(x, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(B)]
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

res = []
for (n, T), g in df.groupby(["tamano", "temperatura"]):
    med, lo, hi = boot_mediana(g["costo"].values)
    res.append({"tamano": n, "temperatura": T, "mediana_boot": med, "IC95_inf": lo, "IC95_sup": hi})
res = pd.DataFrame(res)
print("\nBootstrap por punto de diseño:\n", res.round(2))
res.to_csv(OUT + "bootstrap_doe.csv", index=False)

# Tabla resumen: mejor temperatura (menor mediana) por tamaño
mejores = res.loc[res.groupby("tamano")["mediana_boot"].idxmin()].reset_index(drop=True)
print("\nMejor temperatura por tamaño:\n", mejores.round(2))
mejores.to_csv(OUT + "tabla_optimos.csv", index=False)

# ---------------- 6. Graficas ----------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
m_T = df.groupby("temperatura")["costo"].mean()
ax[0].plot(m_T.index, m_T.values, "o-"); ax[0].set_xlabel("Temperatura T0"); ax[0].set_ylabel("Costo medio")
ax[0].set_title("Efecto principal: temperatura")
m_n = df.groupby("tamano")["costo"].mean()
ax[1].plot(m_n.index, m_n.values, "o-"); ax[1].set_xlabel("Tamaño n"); ax[1].set_ylabel("Costo medio")
ax[1].set_xticks(TAMANOS); ax[1].set_title("Efecto principal: tamaño")
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT + "t3_efectos_principales.png", dpi=150)

plt.figure(figsize=(6.5, 4.5))
for T, g in df.groupby("temperatura"):
    m = g.groupby("tamano")["costo"].mean()
    plt.plot(m.index, m.values, "o-", label=f"T0 = {T}")
plt.xticks(TAMANOS); plt.xlabel("Tamaño n"); plt.ylabel("Costo medio")
plt.title("Interacción temperatura × tamaño"); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(OUT + "t3_interaccion.png", dpi=150)
print("\nListo: resultados en", OUT)