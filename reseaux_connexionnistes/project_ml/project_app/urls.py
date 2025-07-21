from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('confusion_png/', views.confusion_png, name='confusion_png'),
    path('weights_png/', views.weights_png, name='weights_png')
]


# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.index, name='index'),
# ]