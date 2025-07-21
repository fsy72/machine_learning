from django.db import models

# Create your models here.
class PerceptronModel(models.Model):
    w0 = models.FloatField(default=1.0)  # Poids pour le biais
    w1 = models.FloatField(default=1.0)  # Poids pour x1
    w2 = models.FloatField(default=1.0)  # Poids pour x2
    trained = models.BooleanField(default=False)