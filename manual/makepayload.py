# -*- coding: utf-8 -*-


#%% Loading

from gensim.models import Word2Vec
import pandas as pd
import os
import numpy as np
import sys
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from sklearn.cluster import AgglomerativeClustering
from bokeh import palettes
from sqlalchemy import create_engine
import pickle

fast = True
if fast:
    sys.path.append('C:/users/jtay/Desktop/6242/viz proto/bin')
    sys.path.append('C:/users/jtay/Desktop/6242/viz proto')
    from fast_tsne import fast_tsne # O(N) via FFT, see all the comments above...
else:
    raise NotImplementedError('Set fast == True!')


RAND = 4
workers = os.cpu_count() - 2
MOVIE_ID = 'movieId'
TITLE = 'title'
RATING = 'rating'
VECTOR = 'vector'
GENRES = 'genres'
MEAN = 'mean'
COUNT = 'count'
STDDEV = 'std'
X = 'x'
Y = 'y'
CLUSTER = 'cluster'
COLOR = 'color'


model_filename = 'w2v_vs_64_sg_1_hs_1_mc_1_it_4_wn_32_ng_2_all_data_trg_val_tst.gensim'
model = Word2Vec.load(os.path.join('./gensim_models2', model_filename))


with open('metadata.pkl', 'rb') as f:
    dict_metadata = pickle.load(f)

df_movies = pd.read_csv('ml-20m/movies.csv', index_col=MOVIE_ID)

def get_movie_vector(i):
    try:
        return model.wv.get_vector(str(i))
    except KeyError:
        return np.nan

df_movies[VECTOR] = df_movies.index.get_level_values(MOVIE_ID).map(get_movie_vector)
df_movies = df_movies[pd.notnull(df_movies[VECTOR])].copy()
vectors = df_movies[VECTOR].to_numpy()
vectors = np.vstack(vectors)



# %% Run the clustering and tsne



num_clusters = [50,500,1500,5000,10000,len(vectors)]
cluster_list = []
for k in num_clusters:
    clusterer = AgglomerativeClustering(n_clusters=k,linkage='ward',)
    clusterer = clusterer.fit(vectors)
    cluster_list.append(clusterer)
    print(f'k={k} done')




tsne_result = fast_tsne(vectors, seed=RAND, nthreads=workers)



df_movies[X] = tsne_result[:, 0]
df_movies[Y] = tsne_result[:, 1]



print(len(cluster_list))
for clusterer in cluster_list:
    print(clusterer.children_.shape)


#df_movies = bak.copy() # in case something goes pear shaped
for i,clusterer in enumerate(cluster_list):
    df_movies[f'L{i}'] =clusterer.labels_
bak = df_movies.copy()


# %% Get full df
df_movies = bak.copy()
for level in [f'L{i}' for i in range(6)]:
    df_movies[level+'x'] = df_movies.groupby(level)['x'].transform('mean')
    df_movies[level+'y'] = df_movies.groupby(level)['y'].transform('mean')
del df_movies['x']
del df_movies['y']
print(df_movies.head())
del df_movies['vector']

df_output = df_movies.copy()
df_output = df_output.rename(columns={
    'title': 'movie_title',
})
df_output.index.rename('movie_id', inplace=True)
df_output['embedder'] = model_filename


POSTER_URL = 'poster_url'
RUNTIME = 'runtime'
DIRECTOR = 'director'
ACTORS = 'actors'
METASCORE = 'metascore'
IMDB_RATING = 'imdb_rating'
IMDB_VOTES = 'imdb_votes'

df_output[POSTER_URL] = df_output.index.map(lambda x: dict_metadata[x]['Poster']).map(
    lambda x: None if x == 'N/A' else x)
df_output[RUNTIME] = df_output.index.map(
    lambda x: dict_metadata[x]['Runtime']).map(
    lambda x: x.replace(' min', '')).map(
    lambda x: int(x) if x.isdigit() else None)
df_output[DIRECTOR] = df_output.index.map(lambda x: dict_metadata[x]['Director']).map(
lambda x: '|'.join(x.split(', ')))
df_output[ACTORS] = df_output.index.map(lambda x: dict_metadata[x]['Actors']).map(
lambda x: x.replace(', ', '|'))
df_output[METASCORE] = df_output.index.map(lambda x: dict_metadata[x]['Metascore']).map(
    lambda x: int(x) if x.isdigit() else None)
df_output[IMDB_RATING] = df_output.index.map(lambda x: dict_metadata[x]['imdbRating']).map(
    lambda x: float(x) if x != 'N/A' else None)
df_output[IMDB_VOTES] = df_output.index.map(lambda x: dict_metadata[x]['imdbVotes']).map(
    lambda x: int(x.replace(',', '')) if x != 'N/A' else None)

bak = df_output.copy()

# %% Create the dumps we need - output has all of it already
from collections import Counter
df_output = bak.copy()

output = {}
def mostCommon(series,n=10):
    vals = series.tolist()
    tmp = '|'.join(vals)
    tmp = Counter(tmp.split('|')).most_common(10)
    
    return tmp
    

for level in range(5):
    means = df_output.groupby(f'L{level}')[['L5x','L5y','metascore','imdb_rating']].mean()
    means = means.reset_index().rename(columns={f'L{level}':'ID'})
    counts = df_output.groupby(f'L{level}')[['genres','actors']].agg(mostCommon)
    tmp = pd.concat([means,counts],1).rename(columns={'L5x':'x','L5y':'y'}).fillna("null")
    tmp = list(tmp.T.to_dict().values())
    output[level] = tmp.copy()
#    raise
    
    
    
tmp = df_output.copy().fillna("null")
tmp['x'] = tmp.L5x;
tmp['y'] = tmp.L5y;
output[5] = list(tmp.reset_index().rename(columns={'movie_id':'ID'}).T.to_dict().values())

# %% dump! 
import json
with open('payload.json','w') as f:
    json.dump(output,f,allow_nan=False)

