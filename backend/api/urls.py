"""
API URL Routing.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health_check, name='api_health'),
    path('predict/', views.predict_face, name='api_predict'),
    path('hf-predict/', views.hf_predict, name='api_hf_predict'),
    path('metrics/', views.get_metrics, name='api_metrics'),
]
