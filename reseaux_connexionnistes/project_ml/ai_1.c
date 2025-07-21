# toxicite.py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path

# Chargement
model = load_model('Model_Entrainer.h5')
tokenizer = joblib.load('Model_Entrainer.pkl')

# Données test
data = np.loadtxt('test.csv', delimiter=',', skiprows=1)
y_true = data[:, 0].astype(int)
X_test = data[:, 1:]

# Prédictions
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Sortie images
dir_images = Path(__file__).parent / 'project_app' / 'static' / 'images'
dir_images.mkdir(parents=True, exist_ok=True)

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred_class)
disp = ConfusionMatrixDisplay(cm)
fig, ax = plt.subplots()
disp.plot(ax=ax)
fig.savefig(dir_images / 'confusion_matrix.png')
plt.close(fig)

# Évolution des poids (couche 0)
w = model.layers[0].get_weights()[0]
fig, ax = plt.subplots()
ax.plot(np.mean(w, axis=0))
ax.set_title('Évolution moyenne des poids')
fig.savefig(dir_images / 'weights_evolution.png')
plt.close(fig)




import io
import matplotlib.pyplot as plt

# Convertit une figure Matplotlib en PNG en mémoire

def plot_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
    
    
    
from django.shortcuts import render
from django.http import HttpResponse
from .utils import plot_to_png
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Charger une seule fois
dir_base = Path(__file__).resolve().parent.parent
model = load_model(dir_base / 'Model_Entrainer.h5')
tokenizer = joblib.load(dir_base / 'Model_Entrainer.pkl')

# Route principale

def index(request):
    return render(request, 'index.html')

# Génération matrice de confusion
def confusion_png(request):
    data = np.loadtxt(dir_base / 'test.csv', delimiter=',', skiprows=1)
    y_true = data[:, 0].astype(int)
    X_test = data[:, 1:]

    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred_class)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)

    png = plot_to_png(fig)
    return HttpResponse(png, content_type='image/png')

# Génération évolution des poids
def weights_png(request):
    w = model.layers[0].get_weights()[0]
    fig, ax = plt.subplots()
    ax.plot(np.mean(w, axis=0))
    ax.set_title('Évolution moyenne des poids')

    png = plot_to_png(fig)
    return HttpResponse(png, content_type='image/png')
    
    
    
{% load static %}
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Dashboard Toxicité</title>
</head>
<body>
  <h1>Dashboard Toxicité</h1>
  <div>
    <h2>Matrice de confusion</h2>
    <img src="{% url 'confusion_png' %}" alt="Matrice de confusion">
  </div>
  <div>
    <h2>Évolution des poids</h2>
    <img src="{% url 'weights_png' %}" alt="Évolution des poids">
  </div>
</body>
</html>