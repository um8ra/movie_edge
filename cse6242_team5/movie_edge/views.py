from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
from django.template import loader
from django.http import JsonResponse
from pathlib import Path
from gensim.models import Word2Vec
from .forms import SentimentForm
from cse6242_team5.settings import BASE_DIR
from .models import Movie
from django.db.models import Max, Min
import pandas as pd

MOVIE_ID = 'movie_id'
TITLE = 'title'
GENRES = 'genres'
X = 'x'
Y = 'y'

dict_gensim_models = dict()
base_path = Path(BASE_DIR)

# You will probably need to update this
gensim_path = base_path / '..' / 'gensim_models2'
movie_df_path = base_path / '..' / 'ml-20m' / 'movies.csv'

df_movies = pd.read_csv(str(movie_df_path), index_col='movieId', dtype=str)
df_movies.index.rename(MOVIE_ID, inplace=True)


def index(request: HttpRequest) -> HttpResponse:
    embedder = 'w2v_vs_16_sg_1_hs_1_mc_1_it_1_wn_32_ng_2.gensim'  # The only one I've run so far
    movies = Movie.objects.filter(embedder=embedder)
    movies_x_min = movies.aggregate(Min(X))
    movies_x_max = movies.aggregate(Max(X))
    movies_y_min = movies.aggregate(Min(Y))
    movies_y_max = movies.aggregate(Max(Y))

    data = {
        'data': movies,
        'x_min': movies_x_min,
        'x_max': movies_x_max,
        'y_min': movies_y_min,
        'y_max': movies_y_max,
    }

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