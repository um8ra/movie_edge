# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:49:45 2019

@author: Jon
"""

from sklearn import model_selection as ms
import pandas as pd
import pickle
df= pd.read_hdf('binarized.hdf','ratings')
df = df.reset_index()
trg,tst = ms.train_test_split(df,stratify=df.userId,random_state=0,test_size=0.3)
val,tst = ms.train_test_split(tst,stratify=tst.userId,random_state=1,test_size=0.5)
trg = trg.set_index(['userId','movieId'])
tst = tst.set_index(['userId','movieId'])
val = val.set_index(['userId','movieId'])


trg = trg.sort_index()
tst = tst.sort_index()
val = val.sort_index()
trg.to_hdf('binarized.hdf','trg',complib='blosc',complevel=9)
tst.to_hdf('binarized.hdf','tst',complib='blosc',complevel=9)
val.to_hdf('binarized.hdf','val',complib='blosc',complevel=9)

with open('trg.pkl','wb') as f:
    pickle.dump(trg,f)

with open('val.pkl','wb') as f:
    pickle.dump(val,f)


with open('tst.pkl','wb') as f:
    pickle.dump(tst,f)
