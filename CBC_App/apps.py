from django.apps import AppConfig


class CbcAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'CBC_App'
    def ready(self):
        import CBC_App.signals  # noqa