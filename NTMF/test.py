# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 22:12:47 2019

@author: jtay
"""

import numpy as np
import pandas as pd
from fnmtf.loader import load_numpy, save_numpy
from fnmtf.engine import Engine
from fnmtf.common import *
from fnmtf.cod import nmtf_cod
import pandas as pd
from itertools import product
from time import process_time as pt
#from collections import defaultdict
from sklearn.metrics  import roc_auc_score as score


X = load_numpy('trg.npz')
if X is None:
    raise Exception("Unable to open file")
X = X.astype(np.float64)
out = {}

val = pd.read_pickle('val.pkl').reset_index()
rows = val.userId.values
cols = val.movieId.values
vals = val.Liked.values

epsilon = 6
engine = Engine(epsilon=epsilon, parallel=1)
pt = lambda : pd.to_datetime('now')
st = pt()
L = [10,20,30,40,50,60,70,80,90,100]
for k1,k2 in product(L,L):
    params = {'engine': engine, 'X': X, 'k': k1, 'k2': k2, 'seed': 0, 'method': 'nmtf',
    	'technique': 'cod', 'max_iter': 250, 'min_iter': 1, 'epsilon': epsilon,
    	'verbose': False, 'store_history': False, 'store_results': False,
    	'basename': 'ml20m', 'label': 'ml20m'}
    st0 = pt()
    factors, err = nmtf_cod(params)
    print('trained: ',pt()-st0)
    U, S, V = factors
    SV = S.dot(V.T)
    left = U[rows,:]
    right = SV[:,cols].T
    pred = (left*right).sum(1)
    RMSE = np.mean((pred.clip(0,1)-vals)**2)**0.5
    auc = score(vals,pred)
    out[(k1,k2)] = {'rmse':RMSE,'auc':auc}
    print(f'Done case k1:{k1}, k2:{k2}. Cunulative Elapsed:{pt()-st}. Validation RMSE: {RMSE}, AUC: {auc}')
    
    
out = pd.DataFrame(out)    
out.to_csv('NTMF-2.csv')
