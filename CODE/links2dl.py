# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:28:26 2019

@author: jtay
"""

import pandas as pd

links=pd.read_csv('links.csv',index_col=0, dtype = {'imdbId':str})['imdbId']

for i in range(28):
    sub = links.iloc[i*1000:(i+1)*1000]
    sub = 'tt'+sub
    sub = sub.to_frame()
    sub.to_csv(f'f{i+1}.csv')