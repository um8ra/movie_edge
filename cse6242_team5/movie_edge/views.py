from django.shortcuts import render
from django.http import HttpResponse, HttpRequest, JsonResponse
from pathlib import Path
from gensim.models import Word2Vec
from cse6242_team5.settings import BASE_DIR
from .models import Movie, c0, c1, c2, c3, c4
from django.db.models import Max, Min
import pandas as pd
import json
import urllib.parse
import random
from typing import List

# from django.views.decorators.csrf import csrf_exempt

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
LIKE = 'movies_liked'
DISLIKE = 'movies_disliked'
EMBEDDER = 'w2v_vs_64_sg_1_hs_1_mc_1_it_4_wn_32_ng_2_all_data_trg_val_tst.gensim'

db_cols = [MOVIE_ID, MOVIE_TITLE, TITLE, GENRES] + \
          [f'L{i}x' for i in range(6)] + [f'L{i}y' for i in range(6)] + \
          [POSTER_URL, RUNTIME, DIRECTOR, ACTORS, METASCORE, IMDB_RATING, IMDB_VOTES] + ['x', 'y', 'movie_id'] + \
          [f'L{i}' for i in range(6)]

dict_gensim_models = dict()
base_path = Path(BASE_DIR)

# You will probably need to update this
gensim_path = base_path / '..' / 'gensim_models2'
movie_df_path = base_path / '..' / 'ml-20m' / 'movies.csv'

df_movies = pd.read_csv(str(movie_df_path), index_col='movieId', dtype=str)
df_movies.index.rename(MOVIE_ID, inplace=True)


def random_movie_ids(n: int, imdb_votes=10000) -> List[int]:
    # https://stackoverflow.com/questions/1731346/how-to-get-two-random-records-with-django
    all_movie_ids = Movie.objects.filter(embedder=EMBEDDER, imdb_votes__gte=imdb_votes).values_list('id', flat=True)
    random_movies = random.sample(list(all_movie_ids), n)
    return_val = Movie.objects.filter(id__in=random_movies).values_list(MOVIE_ID, flat=True)
    print('Random Movies!')
    print(return_val)
    return list(return_val)


def index(request: HttpRequest) -> HttpResponse:
    movies = Movie.objects.filter(embedder=EMBEDDER).values(*db_cols)

    for movie in movies:
        # This is done since quotes and other junk in the title screws up JSON parsing
        movie[MOVIE_TITLE] = urllib.parse.quote(movie[MOVIE_TITLE])
        movie[DIRECTOR] = urllib.parse.quote(movie[DIRECTOR])
        movie[ACTORS] = urllib.parse.quote(movie[ACTORS])

    cols = ['x', 'y', 'metascore', 'imdb_rating', 'genres', 'actors', 'cluster_id']

    clusters0 = c0.objects.values(*cols)
    clusters1 = c1.objects.values(*cols)
    clusters2 = c2.objects.values(*cols)
    clusters3 = c3.objects.values(*cols)
    clusters4 = c4.objects.values(*cols)
    for clusters in [clusters0, clusters1, clusters2, clusters3, clusters4]:
        for cluster in clusters:
            cluster['actors'] = urllib.parse.quote(cluster['actors'])
            cluster['genres'] = urllib.parse.quote(cluster['genres'])

    movies_x_min = movies.aggregate(Min('x'))
    movies_x_max = movies.aggregate(Max('x'))
    movies_y_min = movies.aggregate(Min('y'))
    movies_y_max = movies.aggregate(Max('y'))

    payload = [list(clusters0), list(clusters1), list(clusters2), list(clusters3), list(clusters4), list(movies)]
    for i, d in enumerate(payload):
        for ele in d:
            ele['ID'] = ele['cluster_id'] if i < 5 else ele['movie_id']

    data = {
        'payload': payload,
        # Since D3 likes to operate on arrays, this decodes movie-id to array position
        'decoder': {m[MOVIE_ID]: i for i, m in enumerate(movies)},
        MOVIE_CHOICES: random_movie_ids(10, 10000),
        **movies_x_min,
        **movies_x_max,
        **movies_y_min,
        **movies_y_max,
    }

    data_json = json.dumps(data)
    return render(request, 'movie_edge/visualization.html',
                  {'table_data': data_json})


# @csrf_exempt
def query_recommendations(request: HttpRequest, topn=10) -> JsonResponse:
    # Making sure model data is fine
    assert gensim_path.is_dir(), "Gensim Directory Not Correct"
    request_data = json.loads(request.body)
    movies_liked = request_data[LIKE]
    movies_liked_int = [int(i) for i in movies_liked]
    movies_disliked = request_data[DISLIKE]
    movies_disliked_int = [int(i) for i in movies_disliked]

    len_movies_liked = len(movies_liked)
    len_movies_disliked = len(movies_disliked)
    if (len_movies_liked + len_movies_disliked) == 0:
        print('No Data: Random')
        response = {MOVIE_CHOICES: random_movie_ids(10)}
        return JsonResponse(response)
    elif len_movies_liked > 0 and len_movies_disliked > 0 and (len_movies_liked + len_movies_disliked) > 10000:
        # Yi and Rocko do stuff here and change the threshold/rules and such
        pass
    else:
        print('Have Data: Calculating...')
        gensim_model_str = EMBEDDER
        print('Likes:')
        print(movies_liked)
        print(df_movies.loc[movies_liked_int])
        print('Dislikes:')
        print(movies_disliked)
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

        # returns List of (movieID, similarity). We only want movieID to return for now.
        response = {MOVIE_CHOICES: [i[0] for i in movies_similar]}
        print(response)
        return JsonResponse(response)
