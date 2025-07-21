from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from keras.models import load_model
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, render_template
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from .forms import ToxicityForm
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour serveur
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

# Configuration des chemins
dir_base = Path(__file__).resolve().parent.parent
model_path = dir_base / 'Model_Entrainer.h5'
scaler_path = dir_base / 'Model_Entrainer.pkl'

# Chargement des modèles avec gestion d'erreurs
def load_trained_model():
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        else:
            print("❌ Modèle non trouvé. Entraînez d'abord le modèle.")
            return None, None
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None, None

def training_model():
    """Fonction corrigée pour l'entraînement du modèle"""
    try:
        # Vérifier l'existence du fichier de données
        data_path = dir_base / "toxicity.csv"
        if not os.path.exists(data_path):
            print(f"❌ Fichier {data_path} non trouvé.")
            return False

        df = pd.read_csv(data_path, sep=';')
        print(f"✅ Données chargées: {df.shape}")

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
        model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compiler le modèle
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Entraîner le modèle
        history = model.fit(X_scaled, y, epochs=100, batch_size=10, verbose=1, validation_split=0.2)

        # Sauvegarder le modèle et le scaler
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        print("✅ Modèle entraîné et sauvegardé.")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        return False

def test_model():
    """Fonction corrigée pour tester le modèle avec sauvegarde du rapport"""
    try:
        # Charger le modèle entraîné et le scaler
        model, scaler = load_trained_model()
        if model is None or scaler is None:
            return False, None, None

        # Charger le fichier de test
        test_path = dir_base / "test.csv"
        if not os.path.exists(test_path):
            print(f"❌ Fichier de test {test_path} non trouvé.")
            return False, None, None

        df_test = pd.read_csv(test_path, sep=';')

        # Séparer les features et la variable cible
        X_test = df_test.iloc[:, :-1].values
        y_raw = df_test.iloc[:, -1].astype(float).values

        # Convertir LC50 en classes binaires : toxique (1) si LC50 <= 3.5
        y_test = (y_raw <= 3.5).astype(int)

        # Normaliser les données de test avec le scaler appris
        X_test_scaled = scaler.transform(X_test)

        # Faire les prédictions
        y_pred = model.predict(X_test_scaled).round().astype(int).flatten()

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        print("✅ Matrice de confusion :\n", cm)

        # Rapport de classification
        class_names = ["Non toxique", "Toxique"]
        report_dict = classification_report(y_test, y_pred, 
                                          target_names=class_names, 
                                          output_dict=True)
        report_text = classification_report(y_test, y_pred, 
                                          target_names=class_names)
        
        print("📋 Rapport de classification :\n", report_text)

        # Créer le répertoire d'images
        script_dir = Path(__file__).resolve().parent
        dir_images = script_dir / 'static' / 'images'
        
        try:
            dir_images.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            print(f"⚠️ Problème avec le répertoire {dir_images}: {e}")
            dir_images = Path.cwd()

        # Sauvegarder les visualisations
        save_classification_results(cm, report_dict, report_text, class_names, dir_images)

        # Résultats détaillés
        df_test["Prévu"] = y_pred
        df_test["Réel"] = y_test
        print("🧾 Résultats ligne par ligne :\n", df_test[["Prévu", "Réel"]].head())

        return True, report_dict, cm

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def formulaire_view(request):
    if request.method == 'POST':
        model, scaler = load_trained_model()
        if model is None or scaler is None:
            return render(request, "index.html", {"error": "Erreur de chargement du modèle"})
        
        form = ToxicityForm(request.POST)
        if form.is_valid():
            # Récupérer les données nettoyées
            cic0 = form.cleaned_data['cic0']
            sm1_dz = form.cleaned_data['sm1_dz']
            gats1i = form.cleaned_data['gats1i']
            ndsch = form.cleaned_data['ndsch']
            ndssc = form.cleaned_data['ndssc']
            mlogp = form.cleaned_data['mlogp']
            
            # Correction : transformer en array 2D
            X = np.array([[cic0, sm1_dz, gats1i, ndsch, ndssc, mlogp]])
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0][0]
            
            result = {
                "classe": "Toxique" if prediction <= 3.5 else "Non toxique",
                "score": round(float(prediction), 2)
            }
            return render(request, "index.html", {"result": result, "form": form})
    else:
        form = ToxicityForm()
    
    return render(request, 'index.html', {'form': form})

def save_classification_results(cm, report_dict, report_text, class_names, dir_images):
    """Sauvegarder tous les résultats de classification"""
    
    # 1. Matrice de confusion stylée
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               square=True, linewidths=0.5)
    plt.title('🔍 Matrice de confusion', fontsize=14, fontweight='bold')
    plt.xlabel('Prédictions', fontweight='bold')
    plt.ylabel('Valeurs réelles', fontweight='bold')

    # 2. Rapport de classification sous forme de heatmap
    plt.subplot(2, 2, 2)
    
    # Créer un DataFrame pour le rapport (exclure 'support' et les moyennes pour la heatmap)
    metrics_data = []
    labels = []
    
    for class_name in class_names:
        if class_name in report_dict:
            metrics_data.append([
                report_dict[class_name]['precision'],
                report_dict[class_name]['recall'],
                report_dict[class_name]['f1-score']
            ])
            labels.append(class_name)
    
    # Ajouter les moyennes
    if 'weighted avg' in report_dict:
        metrics_data.append([
            report_dict['weighted avg']['precision'],
            report_dict['weighted avg']['recall'],
            report_dict['weighted avg']['f1-score']
        ])
        labels.append('Moyenne pondérée')
    
    metrics_df = pd.DataFrame(metrics_data, 
                            columns=['Précision', 'Rappel', 'F1-Score'],
                            index=labels)
    
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', 
               square=False, linewidths=0.5, vmin=0, vmax=1)
    plt.title('📊 Métriques de classification', fontsize=14, fontweight='bold')
    plt.ylabel('')

    # 3. Graphique en barres des métriques
    plt.subplot(2, 2, 3)
    
    x_pos = np.arange(len(class_names))
    width = 0.25
    
    precision_scores = [report_dict[name]['precision'] for name in class_names]
    recall_scores = [report_dict[name]['recall'] for name in class_names]
    f1_scores = [report_dict[name]['f1-score'] for name in class_names]
    
    plt.bar(x_pos - width, precision_scores, width, label='Précision', alpha=0.8)
    plt.bar(x_pos, recall_scores, width, label='Rappel', alpha=0.8)
    plt.bar(x_pos + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('📈 Comparaison des métriques par classe', fontweight='bold')
    plt.xticks(x_pos, class_names)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')

    # 4. Tableau de support (nombre d'échantillons)
    plt.subplot(2, 2, 4)
    
    support_data = [report_dict[name]['support'] for name in class_names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    plt.pie(support_data, labels=class_names, colors=colors, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 10})
    plt.title('📋 Distribution des échantillons\npar classe', fontweight='bold')

    plt.tight_layout()
    
    # Sauvegarder l'image complète
    classification_path = dir_images / 'classification_results.png'
    plt.savefig(classification_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Résultats de classification sauvegardés : {classification_path}")
    plt.close()

    # 5. Rapport de classification textuel stylé
    create_text_report_image(report_text, dir_images)


def create_text_report_image(report_text, dir_images):
    """Créer une image du rapport de classification textuel"""
    
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    
    # Diviser le rapport en lignes
    lines = report_text.strip().split('\n')
    
    # Titre
    plt.text(0.5, 0.95, '📋 Rapport de Classification Détaillé', 
             horizontalalignment='center', verticalalignment='top',
             fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
    
    # Contenu du rapport avec formatage
    y_pos = 0.85
    for i, line in enumerate(lines):
        if line.strip():
            # Formatage spécial pour l'en-tête
            if 'precision' in line and 'recall' in line:
                fontweight = 'bold'
                fontsize = 12
                color = 'darkblue'
            # Formatage pour les lignes de classes
            elif any(cls in line for cls in ['Non toxique', 'Toxique']):
                fontweight = 'normal'
                fontsize = 11
                color = 'darkgreen'
            # Formatage pour les moyennes
            elif 'avg' in line:
                fontweight = 'bold'
                fontsize = 11
                color = 'darkred'
            else:
                fontweight = 'normal'
                fontsize = 10
                color = 'black'
            
            plt.text(0.1, y_pos, line, 
                    horizontalalignment='left', verticalalignment='top',
                    fontfamily='monospace', fontsize=fontsize, 
                    fontweight=fontweight, color=color,
                    transform=plt.gca().transAxes)
            y_pos -= 0.08
    
    plt.tight_layout()
    
    # Sauvegarder le rapport textuel
    report_text_path = dir_images / 'classification_report_text.png'
    plt.savefig(report_text_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Rapport textuel sauvegardé : {report_text_path}")
    plt.close()


# Vues pour Django
def classification_results_png(request):
    """Vue Django pour afficher les résultats de classification"""
    try:
        script_dir = Path(__file__).resolve().parent
        dir_images = script_dir / 'static' / 'images'
        image_path = dir_images / 'classification_results.png'
        
        if image_path.exists():
            with open(image_path, 'rb') as f:
                return HttpResponse(f.read(), content_type='image/png')
        else:
            # Générer une image d'erreur
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, '⚠️ Résultats non disponibles\n\nVeuillez d\'abord:\n1. Entraîner le modèle\n2. Tester le modèle', 
                   ha='center', va='center', transform=ax.transAxes, 
                   fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            ax.axis('off')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plt.close()
            
            return HttpResponse(buffer.getvalue(), content_type='image/png')
            
    except Exception as e:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f'❌ Erreur: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        plt.close()
        
        return HttpResponse(buffer.getvalue(), content_type='image/png')


# def test_model():
#     """Fonction corrigée pour tester le modèle"""
#     try:
#         # Charger le modèle entraîné et le scaler
#         model, scaler = load_trained_model()
#         if model is None or scaler is None:
#             return False

#         # Charger le fichier de test
#         test_path = dir_base / "test.csv"
#         if not os.path.exists(test_path):
#             print(f"❌ Fichier de test {test_path} non trouvé.")
#             return False

#         df_test = pd.read_csv(test_path, sep=';')

#         # Séparer les features et la variable cible
#         X_test = df_test.iloc[:, :-1].values
#         y_raw = df_test.iloc[:, -1].astype(float).values

#         # Convertir LC50 en classes binaires : toxique (1) si LC50 <= 3.5
#         y_test = (y_raw <= 3.5).astype(int)

#         # Normaliser les données de test avec le scaler appris
#         X_test_scaled = scaler.transform(X_test)

#         # Faire les prédictions
#         y_pred = model.predict(X_test_scaled).round().astype(int).flatten()

#         # Affichage de la matrice de confusion
#         cm = confusion_matrix(y_test, y_pred)
#         print("✅ Matrice de confusion :\n", cm)

#         # Affichage du rapport de classification
#         print("📋 Rapport de classification :\n", classification_report(y_test, y_pred))

#         # Créer le répertoire d'images avec vérifications
#         script_dir = Path(__file__).resolve().parent
#         dir_images = script_dir / 'static' / 'images'
        
#         try:
#             dir_images.mkdir(parents=True, exist_ok=True)
#             print(f"📁 Répertoire créé/vérifié : {dir_images}")
            
#             # Tester les permissions
#             test_file = dir_images / '.test_write'
#             test_file.touch()
#             test_file.unlink()
            
#         except (PermissionError, OSError) as e:
#             print(f"⚠️ Problème avec le répertoire {dir_images}: {e}")
#             # Utiliser le répertoire courant comme alternative
#             dir_images = Path.cwd()
#             print(f"📁 Utilisation du répertoire courant : {dir_images}")

#         # Sauvegarder la matrice de confusion
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                    xticklabels=["Non toxique", "Toxique"], 
#                    yticklabels=["Non toxique", "Toxique"])
#         plt.xlabel("Prédit")
#         plt.ylabel("Réel")
#         plt.title("🔍 Matrice de confusion")
#         plt.tight_layout()
        
#         # Chemin du fichier
#         filepath = dir_images / 'confusion_matrix.png'
        
#         try:
#             plt.savefig(filepath, dpi=300, bbox_inches='tight', 
#                        facecolor='white', edgecolor='none')
#             print(f"✅ Image sauvegardée : {filepath}")
            
#             # Vérifier la création du fichier
#             if filepath.exists():
#                 size = filepath.stat().st_size
#                 print(f"📊 Fichier créé - Taille : {size} bytes")
#             else:
#                 print("❌ Le fichier n'a pas été créé malgré l'absence d'erreur")
                
#         except Exception as save_error:
#             print(f"❌ Erreur lors de la sauvegarde : {save_error}")
        
#         plt.show()
#         plt.close('all')

#         # Résultats détaillés ligne par ligne
#         df_test["Prévu"] = y_pred
#         df_test["Réel"] = y_test
#         print("🧾 Résultats ligne par ligne :\n", df_test[["Prévu", "Réel"]].head())

#         return True
        
#     except Exception as e:
#         print(f"❌ Erreur lors du test: {e}")
#         import traceback
#         traceback.print_exc()  # Afficher la stack trace complète
#         return False

def plot_to_png(fig):
    """Convertit une figure matplotlib en PNG pour HTTP response"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def index(request):
    """Vue principale"""
    return render(request, 'index.html')

def confusion_png(request):
    """Génération de la matrice de confusion en PNG"""
    try:
        model, scaler = load_trained_model()
        if model is None or scaler is None:
            # Retourner une image d'erreur ou un message
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'Modèle non chargé\nVeuillez entraîner le modèle', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            png = plot_to_png(fig)
            return HttpResponse(png, content_type='image/png')

        # Charger les données de test
        test_path = dir_base / 'test.csv'
        if not os.path.exists(test_path):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'Fichier test.csv non trouvé', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.axis('off')
            png = plot_to_png(fig)
            return HttpResponse(png, content_type='image/png')

        # Lire les données de test
        df_test = pd.read_csv(test_path, sep=';')
        X_test = df_test.iloc[:, :-1].values
        y_raw = df_test.iloc[:, -1].astype(float).values
        y_true = (y_raw <= 3.5).astype(int)

        # Normaliser et prédire
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled).round().astype(int).flatten()

        # Créer la matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=["Non toxique", "Toxique"],
                   yticklabels=["Non toxique", "Toxique"], ax=ax)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title("Matrice de confusion")

        png = plot_to_png(fig)
        return HttpResponse(png, content_type='image/png')
    
    except Exception as e:
        # En cas d'erreur, retourner une image avec le message d'erreur
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f'Erreur: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        png = plot_to_png(fig)
        return HttpResponse(png, content_type='image/png')

def weights_png(request):
    """Génération de l'évolution des poids en PNG"""
    try:
        model, _ = load_trained_model()
        if model is None:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, 'Modèle non chargé\nVeuillez entraîner le modèle', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=3)
            ax.axis('off')
            png = plot_to_png(fig)
            return HttpResponse(png, content_type='image/png')

        # Récupérer les poids de la première couche
        weights = model.layers[0].get_weights()[0]
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Afficher chaque neurone avec une couleur différente
        for i in range(weights.shape[1]):
            ax.plot(weights[:, i], label=f'Neurone {i+1}', marker='o', linewidth=2)
        
        ax.set_xlabel('Feature')
        ax.set_ylabel('Poids')
        ax.set_title('Poids de la première couche par neurone')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        png = plot_to_png(fig)
        return HttpResponse(png, content_type='image/png')
    
    except Exception as e:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f'Erreur: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        png = plot_to_png(fig)
        return HttpResponse(png, content_type='image/png')

# Fonctions utilitaires pour l'API
def train_model_api(request):
    """API pour entraîner le modèle"""
    if request.method == 'POST':
        success = training_model()
        if success:
            return HttpResponse("✅ Modèle entraîné avec succès", content_type='text/plain')
        else:
            return HttpResponse("❌ Erreur lors de l'entraînement", status=500, content_type='text/plain')
    return HttpResponse("Méthode non autorisée", status=405)

def test_model_api(request):
    """API pour tester le modèle"""
    if request.method == 'POST':
        success = test_model()
        if success:
            return HttpResponse("✅ Test terminé avec succès", content_type='text/plain')
        else:
            return HttpResponse("❌ Erreur lors du test", status=500, content_type='text/plain')
    return HttpResponse("Méthode non autorisée", status=405)