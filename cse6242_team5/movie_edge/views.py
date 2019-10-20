from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
from django.http import JsonResponse
from pathlib import Path
from gensim.models import Word2Vec

dict_gensim_models = dict()
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
    if not gensim_model_path.is_file():
        raise FileNotFoundError

    print(gensim_model_path)
    if gensim_model_str in dict_gensim_models.keys():
        model = dict_gensim_models[gensim_model_str]
    else:
        model = Word2Vec.load(str(gensim_model_path))
        dict_gensim_models[gensim_model_str] = model


    return HttpResponse('I have some recommendations for you!')