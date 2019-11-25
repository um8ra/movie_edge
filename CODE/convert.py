# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:03:12 2019

@author: Jon
"""

import pandas as pd
import numpy as np
import gc,pickle

fname = 'ml-20m/ratings.csv'

ratings = pd.read_csv(fname,index_col=[0,1],dtype={'userId':np.uint32,'movieId':np.uint32,'rating':np.float32,'timestamp':int})
userMedians = ratings.groupby(level=0)['rating'].median()
userMedians.name = 'Medians'
ratings2 = ratings.reset_index().set_index('userId')
del ratings
gc.collect()
ratings2 = ratings2.join(userMedians,how='outer')
ratings2['good'] = ratings2.rating >= ratings2.Medians
ratings2 = ratings2.reset_index().set_index(['userId','movieId'])[['good','timestamp']]
ratings2['good'] = ratings2['good'].astype(np.uint8)
ratings2.columns = ['Liked','Timestamp']
#ratings2.to_csv('binarizedRatings.csv')
with open('binarized.pkl','wb') as f:
    pickle.dump(ratings2,f)
    
ratings2.to_hdf('binarized.hdf','ratings',complevel=9,complib='blosc')
