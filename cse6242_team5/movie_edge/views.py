from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
from django.http import JsonResponse
from pathlib import Path

gensim_models = dict()
current_path = Path('.')

# You will probably need to update this
gensim_path = current_path / '..' / 'gensim_models2'

def index(request):
    return HttpResponse('You may be looking for query_recommendations/<str:gensim_model>/')


def query_recommendations(request: HttpRequest,
                          gensim_model_str: str):
    # Making sure model data is fine
    assert gensim_path.is_dir(), "Gensim Directory Not Correct"

    gensim_model_path = gensim_path / gensim_model_str
    if not gensim_model_path.exists():
        raise FileNotFoundError


    return HttpResponse('I have some recommendations for you!')