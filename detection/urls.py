from django.urls import path
from . import views
from django.shortcuts import render

urlpatterns = [
    path('', views.index),
    path('outcome/', views.evaluate_image),
]