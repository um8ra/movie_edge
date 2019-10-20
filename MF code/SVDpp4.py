# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:07:22 2019

@author: jtay
"""

import pandas as pd
import numpy as np
import surprise
from sklearn.metrics  import roc_auc_score as score
import pickle

pt = lambda : pd.to_datetime('now')
k=[10,15,20,25]
k=20

reader = surprise.Reader(rating_scale=(0, 1))

trg = surprise.Dataset.load_from_df( pd.read_pickle('trg.pkl')['Liked'].reset_index(), reader)
trg = trg.build_full_trainset()
val = surprise.Dataset.load_from_df( pd.read_pickle('val.pkl')['Liked'].reset_index(), reader)
val = val.build_full_trainset().build_testset()

out = {}
for reg in [0.0005]:
    key = (k,reg)
    st = pt()
    print(f'k={k}, reg={reg}, starting...')        
    model = surprise.SVDpp(n_factors=k,reg_all=reg,random_state=0,verbose=True,n_epochs=10) 
#    model = surprise.BaselineOnly()
    model.fit(trg)
    print(f'k={k}, reg={reg}, trained. Elapsed {pt()-st}')
    st = pt()
    res = model.test(val)
    pred = np.array([p.est for p in res])
    y = np.array([p.r_ui for p in res])
    rmse = np.mean((pred.clip(0,1)-y)**2)**0.5
    auc = score(y,pred)
    out[key] = {'rmse':rmse,'auc':auc} 
    print(f'Done case k:{k}, re:{reg}. Test time:{pt()-st}. Validation RMSE: {rmse}, AUC: {auc}')    
    with open(f'SVDpp frozen k20 r{reg}.pkl','wb') as f:
        pickle.dump(model,f)
    
    
    
    

out = pd.DataFrame(out)
out.to_csv(f'SVDpp-4-2.csv')