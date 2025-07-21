from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Données d'exemple
x = np.array([[0], [1], [2], [3], [4]], dtype=float)
y = np.array([[0], [1], [4], [9], [16]], dtype=float)

# Modèle simple
model = Sequential()
model.add(Dense(units=10, input_shape=[1], activation='relu'))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=100, verbose=0)

print("Prédiction pour 5 :", model.predict([5.0]))