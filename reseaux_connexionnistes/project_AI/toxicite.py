import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Lire le fichier avec 908 lignes
df = pd.read_csv("toxicity.csv", sep=';')

# Séparer les features (X) et la variable cible (y)
X = df.iloc[:, :-1].values
y_raw = df.iloc[:, -1].astype(float).values

# Convertir LC50 en classes binaires : toxique (1) si LC50 <= 3.5
y = (y_raw <= 3.5).astype(int)

# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Créer un modèle de réseau de neurones
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraîner le modèle sur **toutes les données**
model.fit(X_scaled, y, epochs=100, batch_size=10, verbose=1)

# Sauvegarder le modèle
model.save("Model_Entrainer.h5")

# Sauvegarder le scaler aussi si tu veux l'utiliser pour normaliser les données de test plus tard
import joblib
joblib.dump(scaler, "Model_Entrainer.pkl")

print("\n✅ Modèle entraîné et sauvegardé.")
