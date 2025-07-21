from django.db import models

class PerceptronModel(models.Model):
    w0 = models.FloatField(default=1.0)  # Poids pour le biais
    w1 = models.FloatField(default=1.0)  # Poids pour x1
    w2 = models.FloatField(default=1.0)  # Poids pour x2
    trained = models.BooleanField(default=False)

    def predict(self, x1, x2):
        activation = self.w0 + self.w1*x1 + self.w2*x2
        return 1 if activation >= 0 else 0