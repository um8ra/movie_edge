from django.urls import path
from . import views

app_name = 'movie_edge'
urlpatterns = [
    path('', views.index, name='index'),
    path('no_graphic/', views.index_no_graphic, name='index_no_graphic'),
    path('query_recommendations/', views.query_recommendations, name='query_recommendations'),
]