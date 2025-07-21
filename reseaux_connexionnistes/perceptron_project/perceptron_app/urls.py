from django.urls import path
from . import views

app_name = 'perceptron_app'
urlpatterns = [
    path('', views.predict_view, name='predict'),
]