# -*- coding: utf-8 -*-
"""
Created on Wen May 21 08:42:22 2025

@author: bebel
"""

# Bibliotecas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# Carregando o dataset (id=891 é o BRFSS2015 com Diabetes_binary desbalanceado)
data = fetch_ucirepo(id=891)
X = data.data.features
y = data.data.targets.values.ravel()

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# Treinamento do modelo
clf = DecisionTreeClassifier(random_state=42, max_depth=5,class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Avaliação
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Matriz de Confusão - Árvore de Decisão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.show()


# Visualização da Árvore
plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=X.columns, class_names=["0", "1"], filled=True, rounded=True, fontsize=8)
plt.title("Árvore de Decisão (Profundidade = 5) - balanceada")
plt.show()
