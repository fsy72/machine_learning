from django.db import models
from django.utils import timezone

class PredictionLog(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    input_features = models.JSONField()
    predicted_class = models.IntegerField()
    confidence = models.FloatField()
    actual_class = models.IntegerField(null=True, blank=True)
    
    class Meta:
        verbose_name = "Log de Prédiction"
        verbose_name_plural = "Logs de Prédictions"
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Prédiction {self.id} - {self.timestamp}"

class ModelTrainingLog(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    accuracy = models.FloatField()
    loss = models.FloatField()
    epochs = models.IntegerField()
    training_duration = models.DurationField()
    notes = models.TextField(blank=True)
    
    class Meta:
        verbose_name = "Log d'Entraînement"
        verbose_name_plural = "Logs d'Entraînements"
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Entraînement {self.id} - Accuracy: {self.accuracy:.3f}"