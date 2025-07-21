from django.apps import AppConfig


class ProjectAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'project_app'
    verbose_name = 'Application de Prédiction de Toxicité'
