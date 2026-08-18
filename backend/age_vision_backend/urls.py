"""
Root URL Configuration for age_vision_backend.
Routes all /api/ traffic to the api application.
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
]
