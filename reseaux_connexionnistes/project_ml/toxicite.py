import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import joblib
from keras.models import load_model
from pathlib import Path

# # Chargement
# model = load_model('Model_Entrainer.h5')
# tokenizer = joblib.load('Model_Entrainer.pkl')

# # Données test
# data = np.loadtxt('test.csv', delimiter=',', skiprows=1)
# y_true = data[:, 0].astype(int)
# X_test = data[:, 1:]

# # Prédictions
# y_pred = model.predict(X_test)
# y_pred_class = np.argmax(y_pred, axis=1)

# # Sortie images
# dir_images = Path(__file__).parent / 'project_app' / 'static' / 'images'
# dir_images.mkdir(parents=True, exist_ok=True)

# # Matrice de confusion
# cm = confusion_matrix(y_true, y_pred_class)
# disp = ConfusionMatrixDisplay(cm)
# fig, ax = plt.subplots()
# disp.plot(ax=ax)
# fig.savefig(dir_images / 'confusion_matrix.png')
# plt.close(fig)

# # Évolution des poids (couche 0)
# w = model.layers[0].get_weights()[0]
# fig, ax = plt.subplots()
# ax.plot(np.mean(w, axis=0))
# ax.set_title('Évolution moyenne des poids')
# fig.savefig(dir_images / 'weights_evolution.png')
# plt.close(fig)


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
