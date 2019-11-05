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

MOVIE_ID = 'movie_id'
MOVIE_TITLE = 'movie_title'
TITLE = 'title'
GENRES = 'genres'
CLUSTER = 'cluster'
COLOR = 'color'
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
    embedder = 'w2v_vs_64_sg_1_hs_1_mc_1_it_4_wn_32_ng_2_all_data_trg_val_tst.gensim'  # The only one I've run so far
    movies = Movie.objects.filter(embedder=embedder).values(MOVIE_ID, MOVIE_TITLE, X, Y, GENRES, CLUSTER)
    palette = palettes.Category20_20
    for movie in movies:
        # This is done since quotes and other junk in the title screws up JSON parsing
        movie[MOVIE_TITLE] = urllib.parse.quote(movie[MOVIE_TITLE])
        movie[COLOR] = palette[movie[CLUSTER]]
    movies_x_min = movies.aggregate(Min(X))
    movies_x_max = movies.aggregate(Max(X))
    movies_y_min = movies.aggregate(Min(Y))
    movies_y_max = movies.aggregate(Max(Y))

    data = {
        'data': list(movies),
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

    return JsonResponse({'mytext': 'I have some recommendations for you!'})
