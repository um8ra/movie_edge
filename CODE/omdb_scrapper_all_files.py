# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:05:30 2019

@author: Jon
"""
import http.client
import pandas as pd
import sys
import pickle
import json

# fnames = sys.argv[1:]

fnames = ['f{}.csv'.format(str(i+1)) for i in range(28)]

conn  = http.client.HTTPConnection('omdbapi.com',80)
key = ''    # patreon key allows 100k API calls / day
            # free key allows only 1k API calls / day

if key == '':
    raise ValueError("Please update API 'key' variable in the python file.")

for fname in fnames:
    print("...loading ", fname)
    out  ={}

    dat = pd.read_csv(fname,index_col=0)['idstr'].to_dict()
    for i,(movieID,imdbID) in enumerate(dat.items()):
        req_s = f'/?apikey={key}&i={imdbID}&plot=full'
        conn.request("GET",req_s)
        res = conn.getresponse()

        print(res.status, res.reason)

        out[movieID] = json.loads(res.read())
        if i % 50 == 0:
            print(i)

    fout = fname.replace('csv','pkl')

    with open(fout,'wb') as f:
        pickle.dump(out,f)
