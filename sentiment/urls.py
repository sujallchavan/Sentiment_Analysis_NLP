from django.urls import path
from .views import sentiment_analysis

urlpatterns = [
    path('', sentiment_analysis, name='sentiment_analysis'),
]
