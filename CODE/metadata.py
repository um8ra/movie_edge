# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:56:45 2019

@author: Jon
"""

import pandas as pd
import pickle

links = pd.read_csv('links.csv')
movies = pd.DataFrame(index=links.movieId,columns = ['len','ok'])

genres = pd.read_csv('movies.csv',index_col=0)['genres']
tags = pd.read_csv('genome-tags.csv',index_col=0)
tagScores = pd.read_csv('genome-scores.csv',index_col=[0,1]).unstack()
tagScores = tagScores.rename(columns = tags['tag'])


d = {}
for n in range(1,29):    
    with open(f'f{n}.pkl','rb') as f:
        dat = pickle.load(f)
        d.update(dat)
    for k,v in dat.items():        
        movies.loc[k,'len'] = len(v)
        movies.loc[k,'ok'] = v['Response'] == 'True'
        d[k]['MovieLensGenres'] = genres.loc[k]
        if k in tagScores.index:
            d[k]['top10Tags'] = tagScores.loc[k].sort_values()['relevance'].tail(10)
        else:
            d[k]['top10Tags'] = None        
        if k % 100 == 0:
            print(k)



#movies = movies['len'].to_frame()
movies.to_csv("metadata-check.csv")

with open('metadata.pkl','wb') as f:
    pickle.dump(d,f)