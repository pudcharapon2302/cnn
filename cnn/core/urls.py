from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("train/start/", views.start_training, name="start_training"),
    path("train/stream/", views.train_stream, name="train_stream"),
    path("predict/", views.predict_digit, name="predict_digit"),
]
