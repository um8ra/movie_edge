from django.shortcuts import render
from django.http import HttpResponse, HttpRequest, JsonResponse
from pathlib import Path
from gensim.models import Word2Vec
from .forms import SentimentForm
from cse6242_team5.settings import BASE_DIR
from .models import Movie
from django.db.models import Max, Min
import pandas as pd
import json
import urllib.parse
from bokeh import palettes
import random
from typing import List

MOVIE_ID = 'movie_id'
MOVIE_TITLE = 'movie_title'
TITLE = 'movie_title'
GENRES = 'genres'
X = 'x'
Y = 'y'
MEAN = 'mean'
COUNT = 'count'
STDDEV = 'std'
CLUSTER = 'cluster'
COLOR = 'color'
POSTER_URL = 'poster_url'
RUNTIME = 'runtime'
DIRECTOR = 'director'
ACTORS = 'actors'
METASCORE = 'metascore'
IMDB_RATING = 'imdb_rating'
IMDB_VOTES = 'imdb_votes'
MOVIE_CHOICES = 'movie_choices'

EMBEDDER = 'w2v_vs_64_sg_1_hs_1_mc_1_it_4_wn_32_ng_2_all_data_trg_val_tst.gensim'

db_cols = [MOVIE_ID, MOVIE_TITLE, TITLE, GENRES, X, Y, MEAN, COUNT, STDDEV, CLUSTER, POSTER_URL, RUNTIME, DIRECTOR,
           ACTORS, METASCORE, IMDB_RATING, IMDB_VOTES]

dict_gensim_models = dict()
base_path = Path(BASE_DIR)

# You will probably need to update this
gensim_path = base_path / '..' / 'gensim_models2'
movie_df_path = base_path / '..' / 'ml-20m' / 'movies.csv'

df_movies = pd.read_csv(str(movie_df_path), index_col='movieId', dtype=str)
df_movies.index.rename(MOVIE_ID, inplace=True)


def random_movie_ids(n: int) -> List[int]:
    # https://stackoverflow.com/questions/1731346/how-to-get-two-random-records-with-django
    all_movie_ids = Movie.objects.filter(embedder=EMBEDDER).values_list('id', flat=True)
    random_movies = random.sample(list(all_movie_ids), n)
    return_val = Movie.objects.filter(id__in=random_movies).values_list(MOVIE_ID, flat=True)
    print('Random Movies!')
    print(return_val)
    return list(return_val)

def index(request: HttpRequest) -> HttpResponse:
    movies = Movie.objects.filter(embedder=EMBEDDER).values(*db_cols)
    palette = palettes.Category20_20
    for movie in movies:
        # This is done since quotes and other junk in the title screws up JSON parsing
        movie[MOVIE_TITLE] = urllib.parse.quote(movie[MOVIE_TITLE])
        movie[DIRECTOR] = urllib.parse.quote(movie[DIRECTOR])
        movie[ACTORS] = urllib.parse.quote(movie[ACTORS])
        movie[COLOR] = palette[movie[CLUSTER]]
    movies_x_min = movies.aggregate(Min(X))
    movies_x_max = movies.aggregate(Max(X))
    movies_y_min = movies.aggregate(Min(Y))
    movies_y_max = movies.aggregate(Max(Y))

    frequently_rated_movies = movies.filter(imdb_votes__gte=10000)
    len_movies = len(frequently_rated_movies)
    randoms = [random.randint(0, len_movies - 1) for _ in range(9)]
    serendipity_movies = [frequently_rated_movies[i][MOVIE_ID] for i in randoms]
    print(serendipity_movies)
    data = {
        'data': list(movies),
        # Since D3 likes to operate on arrays, this decodes movie-id to array position
        'decoder': {m[MOVIE_ID]: i for i, m in enumerate(movies)},
        'random_nine': serendipity_movies,
        **movies_x_min,
        **movies_x_max,
        **movies_y_min,
        **movies_y_max,
    }

    data_json = json.dumps(data)
    return render(request, 'movie_edge/visualization.html',
                  {'table_data': data_json,
                   'form': SentimentForm})


def sentiment_form(request: HttpRequest) -> HttpResponse:
    return render(request, 'movie_edge/sentiment_form.html',
                  {'form': SentimentForm()})


def query_recommendations(request: HttpRequest, topn=5) -> JsonResponse:
    # Making sure model data is fine
    assert gensim_path.is_dir(), "Gensim Directory Not Correct"
    print('Hello there, General Kenobi!')
    if request.method == 'POST':
        form = SentimentForm(request.POST)

        # This isn't used since the form validation fails on the fetch() request in the JS
        # However, Yi, this is good scaffolding for you to use.
        # I need this function to return 9 movie recommendations
        if form.is_valid():
            print(form.cleaned_data)
            # gensim_model_str = form.cleaned_data['gensim_model']
            gensim_model_str = EMBEDDER
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

    return JsonResponse({MOVIE_CHOICES: random_movie_ids(9)})
