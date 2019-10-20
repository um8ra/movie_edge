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

modelfile = 'NMTF dump k1=80 k2=40.npz'

modelfile = np.load(modelfile)
U = modelfile['U']
V = modelfile['V']
S = modelfile['S']


movieVectors = V.T
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



