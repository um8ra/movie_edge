from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
from django.template import loader
from django.http import JsonResponse
from pathlib import Path
from gensim.models import Word2Vec
from .forms import SentimentForm
from cse6242_team5.settings import BASE_DIR
print(BASE_DIR)


dict_gensim_models = dict()
base_path = Path(BASE_DIR)
print(base_path)

# You will probably need to update this
gensim_path = base_path / '..' / 'gensim_models2'

def index(request):
    return HttpResponse('You may be looking for query_recommendations/<str:gensim_model>/')

def sentiment_form(request):
    # template = loader.get_template('movie_edge/sentiment_form.html')

    return render(request, 'movie_edge/sentiment_form.html',
                  {'form': SentimentForm()})


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