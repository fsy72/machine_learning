from django.shortcuts import render, redirect
from .models import PerceptronModel
import random

def train_perceptron():
    # Données d'entraînement (OU logique)
    training_data = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1)
    ]
    
    # Créer ou récupérer le modèle
    model, created = PerceptronModel.objects.get_or_create(id=1)
    
    # Initialisation aléatoire des poids
    model.w0 = random.uniform(-1, 1)
    model.w1 = random.uniform(-1, 1)
    model.w2 = random.uniform(-1, 1)
    
    # Paramètres d'apprentissage
    learning_rate = 0.1
    epochs = 100
    
    # Algorithme du perceptron
    for _ in range(epochs):
        for x1, x2, target in training_data:
            # Calculer la prédiction actuelle
            prediction = model.predict(x1, x2)
            
            # Mettre à jour les poids
            model.w0 += learning_rate * (target - prediction) * 1  # *1 pour le biais
            model.w1 += learning_rate * (target - prediction) * x1
            model.w2 += learning_rate * (target - prediction) * x2
    
    model.trained = True
    model.save()

def predict_view(request):
    model, _ = PerceptronModel.objects.get_or_create(id=1)
    
    # Entraîner si pas encore fait
    if not model.trained:
        train_perceptron()
        model.refresh_from_db()
    
    result = None
    if request.method == 'POST':
        try:
            x1 = int(request.POST.get('x1', 0))
            x2 = int(request.POST.get('x2', 0))
            result = model.predict(x1, x2)
        except ValueError:
            result = "Erreur: Entrez 0 ou 1"
    
    return render(request, 'perceptron_app/predict.html', {
        'result': result,
        'weights': [model.w0, model.w1, model.w2]
    })