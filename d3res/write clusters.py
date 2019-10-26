# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 09:47:16 2019

@author: jtay
"""

#fast tsne code

import sys; sys.path.append('../')
from fast_tsne import fast_tsne
from gensim.models import Word2Vec
import sklearn.decomposition as decomp
import pandas as pd
import numpy as np
import hdbscan
import sklearn.cluster as cluster
import json

#X = np.random.randn(1000, 50)
#Z = fast_tsne(X, perplexity = 30)

#Yi's loader
model_path='w2v_vs_64_sg_1_hs_1_mc_1_it_4_wn_32_ng_2.gensim'

model  = Word2Vec.load(model_path)
movie2Vec = model.wv
movie_IDs = list(movie2Vec.vocab.keys())
movie_vectors_dict = dict()
for _movie_ID in movie_IDs:
    movie_vectors_dict[_movie_ID] = movie2Vec[_movie_ID]

movie_df = pd.DataFrame(movie_vectors_dict).T
X = movie_df.values

metadata = pd.read_pickle('./metadata.pkl')

# tsne vs pca
#perplexity = 30
#Z_tsne =  fast_tsne(X, perplexity = perplexity)
#Z_pca = decomp.PCA(n_components=2).fit_transform(X)
#Z_tsne = pd.DataFrame(Z_tsne)
#Z_tsne.plot(kind='scatter',x=0,y=1,title=f't-SNE, perplexity={perplexity}')
#
#
#Z_pca = pd.DataFrame(Z_pca)
#Z_pca.plot(kind='scatter',x=0,y=1,title='PCA')

# coordinates
perplexity = 30
Z_tsne =  fast_tsne(X, perplexity = perplexity)
Z_pca = decomp.PCA(n_components=2).fit_transform(X)
Z_tsne = pd.DataFrame(Z_tsne)
Z_tsne.columns =['x','y']
Z_tsne.index = movie_df.index
Z_tsne.index.name = 'movieID'
Z_tsne =Z_tsne.reset_index()
Z_tsne.movieID= Z_tsne.movieID.astype(int)
#i2id = Z_tsne.movieID.to_dict()

# hdbscan has lots of outliers
#clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
#clusterer.fit(X)
#print(sum(clusterer.labels_==-1))

# WARD CLUSTERING
#clusterer = cluster.AgglomerativeClustering()
ns = set(np.logspace(0,10,50,base=3).astype(int))
ns = sorted(list(ns))
ns = [n for n in ns if n < X.shape[0]]
for n in ns:
    clusterer = cluster.AgglomerativeClustering(n_clusters=n)
    labels = clusterer.fit_predict(X)
    h = f'c{n}'
    print(h)
    Z_tsne[h]=labels

Z_tsne.to_pickle('clustered.pkl')


#to json
for col in [f'c{n}' for n in ns]:
    tmp = Z_tsne[['movieID','x','y',col]]
    tmp=tmp.set_index('movieID')
    tmp['x'] = tmp.groupby(col)['x'].transform('mean')
    tmp['y']= tmp.groupby(col)['y'].transform('mean')
    tmp = tmp.reset_index()[['x','y','movieID']]
    tmp.movieID = tmp.movieID.astype(str)
    out = [x[1].to_dict() for x in tmp.iterrows()]
    with open(f'{col}.json','w',encoding='utf-8') as f:
        json.dump(out,f)
    
#    new_x = tmp.groupby(col)['x'].agg('mean')
#    new_y = tmp.groupby(col)['y'].agg('mean')
#    new_id = tmp.groupby(col)['movieID'].agg(lambda x: '|'.join(x.astype(str)))
#    tmp=pd.DataFrame({'x':new_x,'y':new_y,'ids':new_id})
    
