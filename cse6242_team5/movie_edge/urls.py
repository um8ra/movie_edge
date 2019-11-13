from django.urls import path
from . import views

app_name = 'movie_edge'
urlpatterns = [
    path('', views.index, name='index'),
    path('query_recommendations/', views.query_recommendations, name='query_recommendations'),
]