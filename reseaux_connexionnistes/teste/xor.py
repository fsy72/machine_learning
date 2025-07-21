import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model

# Exemple de données
training_data = np.array([[0, 0],
                          [0, 1],
                          [1, 0],
                          [1, 1]], dtype=float)

target_data = np.array([[0], [1], [1], [0]], dtype=float)  # Exemple XOR

# Construction du modèle
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

# Entraînement
model.fit(training_data, target_data, epochs=1000)

# Évaluation
scores = model.evaluate(training_data, target_data)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Prédictions
print(model.predict(training_data).round())

# Résumé du modèle
model.summary()

# Sauvegarde de l'architecture du modèle en image
plot_model(model, to_file='model.png', show_shapes=True)
