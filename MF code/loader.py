# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:55:31 2019

@author: jtay
"""

import pickle
import bz2
import surprise
import numpy as np
import sklearn.neighbors as neigh
from scipy.spatial.distance import cosine

# Load metadata

with open('metadata.pkl','rb') as f:
    metadata = pickle.load(f)

modelfile = 'SVD frozen k20 r0.0001.pkl.bz2'
#modelfile = 'SVDpp frozen k20 r0.0.pkl.bz2'

with bz2.open(modelfile) as f:
    model=pickle.load(f)
    
trainset = model.trainset

n_movies = max(metadata.keys())+1
n_latent = model.qi.shape[1]
movieVectors = np.ones((n_movies,n_latent))*999

for i,iid in enumerate(metadata.keys()):
    try:
        inner = trainset.to_inner_iid(iid)
        movieVectors[iid] = model.qi[inner]
    except ValueError:
        pass
    if i % 1000 ==0:
        print(i)


# %%
        
query = 2571 #260
#targets = [1196, 1210,  2628, 5378, 33493]
targets = [6365, 6934,  27660]

print(f'Query id={query}, movie={metadata[query]["Title"]}, Movielens Genre ={metadata[query]["MovieLensGenres"]}')
for t in targets:
    sim = 1-cosine(movieVectors[query],movieVectors[t])
    print(f'Similarity={sim}, id={t}, movie={metadata[t]["Title"]}, Movielens Genre ={metadata[t]["MovieLensGenres"]}')
n=10
print(f'Top {n} matches')
nn = neigh.NearestNeighbors(n_neighbors=n+1,metric='cosine')
nn.fit(movieVectors)
queryV = movieVectors[query].reshape(1,-1)
dist, ind = nn.kneighbors(queryV)
ind = ind[0]
dist = dist[0]
for j,(i,s) in enumerate(zip(ind,dist)):
    if i == query: 
        continue
    s = 1-s
    print(f'Match {j}, Similarity={s}, id={i}, movie={metadata[i]["Title"]}, Movielens Genre ={metadata[i]["MovieLensGenres"]}')



