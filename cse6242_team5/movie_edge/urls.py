from django.urls import path
from . import views

app_name = 'movie_edge'
urlpatterns = [
    path('', views.index, name='index'),
    path('sentiment_form/', views.sentiment_form, name='sentiment_form'),
    path('query_recommendations/', views.query_recommendations, name='query_recommendations'),
]