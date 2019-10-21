from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
from django.template import loader
from django.http import JsonResponse
from pathlib import Path
from gensim.models import Word2Vec
from .forms import SentimentForm
from cse6242_team5.settings import BASE_DIR
import pandas as pd

MOVIE_ID = 'movieId'
TITLE = 'title'
GENRES = 'genres'

dict_gensim_models = dict()
base_path = Path(BASE_DIR)

# You will probably need to update this
gensim_path = base_path / '..' / 'gensim_models2'
movie_df_path = base_path / '..' / 'ml-20m' / 'movies.csv'

df_movies = pd.read_csv(str(movie_df_path), index_col=MOVIE_ID, dtype=str)
print(df_movies.dtypes)
def index(request: HttpRequest) -> HttpResponse:
    return HttpResponse('You may be looking for query_recommendations/<str:gensim_model>/')


def sentiment_form(request: HttpRequest) -> HttpResponse:
    return render(request, 'movie_edge/sentiment_form.html',
                  {'form': SentimentForm()})


def query_recommendations(request: HttpRequest, topn=5) -> HttpResponse:
    # Making sure model data is fine
    assert gensim_path.is_dir(), "Gensim Directory Not Correct"

    if request.method == 'GET':
        form = SentimentForm(request.GET)

        if form.is_valid():
            print(form.cleaned_data)
            gensim_model_str = form.cleaned_data['gensim_model']
            movies_liked = form.cleaned_data['likes'].replace(' ', '').split(',')
            movies_liked_int = [int(i) for i in movies_liked]
            movies_disliked = form.cleaned_data['dislikes'].replace(' ', '').split(',')
            movies_disliked_int = [int(i) for i in movies_disliked]
            print(movies_liked)
            print('Likes:')
            print(df_movies.loc[movies_liked_int])
            print('Dislikes:')
            print(df_movies.loc[movies_disliked_int])

            gensim_model_path = gensim_path / gensim_model_str
            if not gensim_model_path.is_file():
                raise FileNotFoundError

            if gensim_model_str in dict_gensim_models.keys():
                model = dict_gensim_models[gensim_model_str]
            else:
                model = Word2Vec.load(str(gensim_model_path))
                dict_gensim_models[gensim_model_str] = model

            movies_similar = model.most_similar(positive=movies_liked,
                                                negative=movies_disliked,
                                                topn=topn)

            print('Similar: ')
            print(df_movies.loc[[int(i[0]) for i in movies_similar]])


    return HttpResponse('I have some recommendations for you!')