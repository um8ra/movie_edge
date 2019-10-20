from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('query_recommendations/<str:gensim_model_str>/', views.query_recommendations),
]