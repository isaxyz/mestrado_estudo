# -*- coding: utf-8 -*-
"""
Created on Mon Aug 08 19:13:06 2025

@author: bebel
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, mannwhitneyu


# Carregar os dados
df = pd.read_csv("dados_simulacao_final.csv", sep=";", encoding="ISO-8859-1", decimal=",")

# teste para ver se os dados estao sendo consultados corretamente
print(df.head())
print(df.columns.tolist())

# histograma do ripple
sns.histplot(df['Ripple'], kde=True)
plt.title("Distribuição do Ripple")
plt.show()


# grafico de correlacao das variaveis
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlação entre variáveis")
plt.show()

# Definir X - tudo menos o ripple e y
X = df.drop(columns=["Ripple"])
y = df["Ripple"]


# treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# modelagem
modelos = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=0.5),
    "Lasso Regression": Lasso(alpha=0.01)
}


resultados = {}

for nome, modelo in modelos.items():
    # treinar modelos
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    # calculo das metricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # resultados
    resultados[nome] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE (%)": mape
    }

# Transformar em DataFrame para visualização
df_resultados = pd.DataFrame(resultados).T
print(df_resultados)

# comparacao entre os 3 modelos
plt.figure(figsize=(14, 5))

for i, (nome, modelo) in enumerate(modelos.items(), 1):
    plt.subplot(1, 3, i)
    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)  # calcular R² para o modelo
    
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Previsões")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             color='red', linestyle='--', label="Linha Ideal (y=x)")
    
    plt.title(f"{nome}\nR² = {r2:.3f}")
    plt.xlabel("Valores Reais (Ripple)")
    plt.ylabel("Valores Previstos (Ripple)")
    plt.legend()

plt.suptitle("Comparação: Valores Reais vs Previstos", fontsize=14)
plt.tight_layout()
plt.show()

# residuos para cada modelo

residuos = {}

for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_test)
    res = y_test.values - y_pred
    residuos[nome] = res


# teste para saber se os dados sao normais ou nao (Shapiro-Wilk)

print("\nTeste de Normalidade (Shapiro-Wilk):")
for nome, res in residuos.items():
    stat, p = shapiro(res)
    print(f"{nome}: stat={stat:.4f}, p={p:.5f}")


# teste de Mann-Whitney - dados nao parametricos nao pareado

print("\nMann-Whitney U test (não pareado, 2 a 2):")
pares = [("Linear Regression", "Ridge Regression"),
         ("Linear Regression", "Lasso Regression"),
         ("Ridge Regression", "Lasso Regression")]

for m1, m2 in pares:
    stat, p = mannwhitneyu(residuos[m1], residuos[m2])
    print(f"{m1} vs {m2}: stat={stat:.3f}, p={p:.5f}")
