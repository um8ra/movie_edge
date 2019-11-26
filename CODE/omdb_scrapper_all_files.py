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

fname = 'ml-20m/links.csv'

print("...loading ", fname)
out = {}

dat = pd.read_csv(fname, index_col=0)
dat['idstr'] = dat['imdbId'].apply(lambda x: 'tt'+('0000000'+str(x))[-7:])

dat = dat['idstr'].to_frame()
# print(dat.tail(10))

# $1/month patreon membership grants 100k API requests / day
# $5/month patreon membership grants 250k API requests / day
# Free key only allows only 1k API calls / day
# There are 27k+ movies. Will need to pay for $5/month patreon membership
key = ''    

if key == '':
    raise ValueError("Please update API 'key' variable in the python file.")

conn  = http.client.HTTPConnection('omdbapi.com',80)

# dat = dat.head(10)    # uncomment to test for just 10 movies
dat = dat['idstr'].to_dict()
for i, (movieID, imdbID) in enumerate(dat.items()):
    req_s = f'/?apikey={key}&i={imdbID}&plot=full'
    conn.request("GET",req_s)
    res = conn.getresponse()

    print(movieID, imdbID, res.status, res.reason)

    out[movieID] = json.loads(res.read())

fout = 'metadata.pkl'

with open(fout,'wb') as f:
    pickle.dump(out,f)
