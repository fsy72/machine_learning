import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le modèle entraîné et le scaler
model = load_model("Model_Entrainer.h5")
scaler = joblib.load("Model_Entrainer.pkl")

# Charger le fichier de test
df_test = pd.read_csv("test.csv", sep=';')

# Séparer les features et la variable cible
X_test = df_test.iloc[:, :-1].values
y_raw = df_test.iloc[:, -1].astype(float).values

# Convertir LC50 en classes binaires : toxique (1) si LC50 <= 3.5
y_test = (y_raw <= 3.5).astype(int)

# Normaliser les données de test avec le scaler appris
X_test_scaled = scaler.transform(X_test)

# Faire les prédictions
y_pred = model.predict(X_test_scaled).round().astype(int).flatten()

# Affichage de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("\n✅ Matrice de confusion :\n", cm)

# Affichage du rapport de classification
print("\n📋 Rapport de classification :\n", classification_report(y_test, y_pred))

# ➕ Affichage graphique de la matrice de confusion
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non toxique", "Toxique"], yticklabels=["Non toxique", "Toxique"])
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("🔍 Matrice de confusion")
plt.tight_layout()
plt.show()

# Résultats détaillés ligne par ligne
df_test["Prévu"] = y_pred
df_test["Réel"] = y_test
print("\n🧾 Résultats ligne par ligne :\n", df_test[["Prévu", "Réel"]].head())
