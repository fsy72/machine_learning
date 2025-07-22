# urls.py - Configuration des URLs pour l'application Django

from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

# URLs de l'application
urlpatterns = [
    # Page principale
    path('', views.index, name='index'),
    path('home/', views.index, name='home'),
    
    # Génération d'images (graphiques)
    path('confusion-matrix.png', views.confusion_png, name='confusion_matrix_png'),
    path('weights-evolution.png', views.weights_png, name='weights_evolution_png'),
    
    path('classification-results.png', views.classification_results_png, name='classification_results_png'),
    
    # APIs pour l'entraînement et le test
    path('api/train/', views.train_model_api, name='train_model_api'),
    path('api/test/', views.test_model_api, name='test_model_api'),
    path('api/predict/', views.predict_api, name='predict_api'),
    
    # URLs alternatives (optionnelles)
    path('matrix/', views.confusion_png, name='confusion_matrix'),
    path('weights/', views.weights_png, name='weights_visualization'),
]

# Configuration pour servir les fichiers statiques en développement
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


# Exemple d'usage dans les templates HTML :
"""
<!-- Dans votre template HTML -->
<!DOCTYPE html>
<html>
<head>
    <title>Analyse de Toxicité</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .button { 
            background-color: #007bff; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer; 
            margin: 5px;
            text-decoration: none;
            display: inline-block;
        }
        .button:hover { background-color: #0056b3; }
        .image-container { margin: 20px 0; text-align: center; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Analyse de Toxicité - Modèle ML</h1>
        
        <!-- Boutons d'action -->
        <div class="actions">
            <h2>Actions</h2>
            <button onclick="trainModel()" class="button">🏋️ Entraîner le Modèle</button>
            <button onclick="testModel()" class="button">🧪 Tester le Modèle</button>
            <button onclick="refreshImages()" class="button">🔄 Actualiser les Graphiques</button>
        </div>
        
        <!-- Zone de statut -->
        <div id="status"></div>
        
        <!-- Graphiques -->
        <div class="image-container">
            <h2>📊 Matrice de Confusion</h2>
            <img id="confusion-matrix" src="{% url 'confusion_matrix_png' %}" alt="Matrice de confusion" />
        </div>
        
        <div class="image-container">
            <h2>⚖️ Évolution des Poids</h2>
            <img id="weights-evolution" src="{% url 'weights_evolution_png' %}" alt="Évolution des poids" />
        </div>
    </div>

    <script>
        function showStatus(message, isError = false) {
            const statusDiv = document.getElementById('status');
            statusDiv.innerHTML = `<div class="status ${isError ? 'error' : 'success'}">${message}</div>`;
            setTimeout(() => statusDiv.innerHTML = '', 5000);
        }

        async function trainModel() {
            showStatus('🏋️ Entraînement en cours...');
            try {
                const response = await fetch('{% url "train_model_api" %}', {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': '{{ csrf_token }}',
                        'Content-Type': 'application/json'
                    }
                });
                const text = await response.text();
                showStatus(text, !response.ok);
                if (response.ok) refreshImages();
            } catch (error) {
                showStatus('❌ Erreur réseau: ' + error.message, true);
            }
        }

        async function testModel() {
            showStatus('🧪 Test en cours...');
            try {
                const response = await fetch('{% url "test_model_api" %}', {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': '{{ csrf_token }}',
                        'Content-Type': 'application/json'
                    }
                });
                const text = await response.text();
                showStatus(text, !response.ok);
                if (response.ok) refreshImages();
            } catch (error) {
                showStatus('❌ Erreur réseau: ' + error.message, true);
            }
        }

        function refreshImages() {
            const timestamp = new Date().getTime();
            document.getElementById('confusion-matrix').src = '{% url "confusion_matrix_png" %}?t=' + timestamp;
            document.getElementById('weights-evolution').src = '{% url "weights_evolution_png" %}?t=' + timestamp;
            showStatus('🔄 Graphiques actualisés');
        }

        // Auto-refresh des images toutes les 30 secondes
        setInterval(refreshImages, 30000);
    </script>
</body>
</html>
"""