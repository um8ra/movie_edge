import numpy as np
import pandas as pd
from gensim.models import Word2Vec
# from sklearn.model_selection import ParameterGrid
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import roc_auc_score

import pickle
# import h5py
import os
import sys
from datetime import datetime
# import logging
# import shutil

# COLUMNS
LIKED = 'Liked'
MOVIE_ID = 'movieId'
USER_ID = 'userId'
TIMESTAMP = 'Timestamp'
TITLE = 'title'
GENRE = 'genres'

## Note that dot product may not actually make sense
## cosine similarity is between -1 and 1, with 1 being more similar.
## If we do dot product, it would have norm(vec1) and norm(vec2)
## as magnitute multiplied on top of the cos sim.
## The euclidean distance would be further with large norms.
SCORE_METHOD = 0 # sklearn pairwise metric if 0, else dot product
PAIRWISE_METRIC = cosine_similarity # euclidean_distances

# MODEL_NAME = 'w2v_vs_16_sg_1_hs_1_mc_1_it_1_wn_32_ng_2'
MODEL_NAME = 'w2v_vs_16_sg_1_hs_1_mc_5_it_1_wn_32_ng_2'
## if evaluating AUROC on trg, COMMENT out df_trg = df_trg[df_trg[LIKED] == 1]
EVAL_DATASET = 'val' # 'trg' or 'val'


def run_validation_metrics():

    ## --------------------------------------------------------------
    ## Used helper functions for validation purpose
    ## --------------------------------------------------------------
    def transform_df(_df):
        _df = _df.drop([TIMESTAMP], axis=1)
        _df[MOVIE_ID] = _df.index.get_level_values(MOVIE_ID).astype(str)
        return _df

    df_movies = pd.read_csv('ml-20m/movies.csv', index_col=MOVIE_ID)
    print('There are totally {} movies'.format(df_movies.index.size)) #
    print()
    ## --------------------------------------------------------------


    ## --------------------------------------------------------------
    ## Unused helper functions for validation purpose
    ## --------------------------------------------------------------
    def show_synonyms(df_movies, model, search_str, num_synonyms):
        synonym_list = list()
        movie_index = df_movies[df_movies[TITLE].str.match(search_str)]
        print(movie_index)
        for mi in movie_index.index:
            synonym_list.extend([(i, df_movies.loc[int(i[0])][TITLE]) for i in 
                                 list(model.findSynonyms(str(mi), num_synonyms))])
        return synonym_list


    def show_synonyms(df_movies, model, search_str, num_synonyms, verbose=True):
        synonym_list = list()
        movie_index = df_movies[df_movies[TITLE].str.match(search_str)]
        for mi in movie_index.index:
            synonym_list.extend([(i, df_movies.loc[int(i[0])][TITLE]) for i in 
                                 list(model.wv.most_similar(str(mi), topn=num_synonyms))])
        cosine_similarity = pd.Series([i[0][1] for i in synonym_list])
        mean = cosine_similarity.mean()
        stddev = cosine_similarity.std()
        if verbose:
            print(movie_index)
            print('Mean: {} \t StdDev: {}'.format(mean, stddev))
        return synonym_list, mean, stddev

    def movie2vec(df_movies, model, search_str, verbose=True):
        movie_list = list()
        movie_index = df_movies[df_movies[TITLE].str.match(search_str)]
        if verbvose:
            print(movie_index)
        for mi in movie_index.index:
            movie_list.extend([(mi, model.wv[str(mi)])])
        return movie_list
    ## --------------------------------------------------------------


    ## --------------------------------------------------------------
    ## Load pre-trained Word2Vec Model into keyedvectors object
    ## then convert to dict and pandas.DataFrame with movie_IDs index
    ## --------------------------------------------------------------

    myModel = Word2Vec.load('Model/gensim/{}.gensim'.format(MODEL_NAME))
    movie2vec_kv = myModel.wv
    # print(type(movie2vec_kv))     # <class 'gensim.models.keyedvectors.Word2VecKeyedVectors'>
    # print(type(myModel.wv.vocab)) # dict, values are gensim.models.keyedvectors.Vocab object
    # print(list(myModel.wv.vocab.keys())[0:10])    
    # print(myModel.wv.vocab['589'])
    # print()
    # print(type(myModel.wv.index2word)) # list
    # print(myModel.wv.index2word[0:10])

    movie2vec_dict = dict()
    for i, _movie_ID in enumerate(myModel.wv.vocab):
        movie2vec_dict[_movie_ID] = myModel.wv[_movie_ID]

    print('There are totally {} movie embeddings'.format(
        # len(myModel.wv.vocab)))
        # len(myModel.wv.index2word)))
        len(movie2vec_dict)))
    print()

    movie_IDs = list(movie2vec_dict.keys())
    movie_vectors = np.array(list(movie2vec_dict.values()))
    movie_vectors_df = pd.DataFrame(movie_vectors, index=movie_IDs)

    # print(myModel.wv['589'])
    # print(myModel.wv.get_vector('589'))
    # print(myModel.wv.word_vec('589'))
    # print(movie2vec_dict['589'])

    # # test_movie_vectors = movie2vec(df_movies, myModel, 'Saving Private Ryan') # '2028'
    # test_movie_vectors = movie2vec(df_movies, myModel, 'Terminator 2: Judgment Day') # '589'
    # print(test_movie_vectors)
    # print()
    ## --------------------------------------------------------------


    ## ********************************************************************
    ## skip the next 2 steps if user_vectors_dict is saved to pickle
    ##
    ## --------------------------------------------------------------------
    ## Load training data
    ## --------------------------------------------------------------------
    df_trg = pd.read_hdf('Ratings/binarized.hdf', key='trg')
    df_trg = df_trg[df_trg[LIKED] == 1] # comment out if evaluating AUROC
    # df_trg = df_trg.head(50000) # todo comment out for production
    df_trg = transform_df(df_trg)
    # df_trg = df_trg.head(10) # quick check on 10 movies

    df_trg_gb = df_trg.groupby([USER_ID])
    dict_groups_trg = {k: list(v[MOVIE_ID]) for k, v in df_trg_gb}
    ## --------------------------------------------------------------------


    ## --------------------------------------------------------------------
    ## Calculate each user vector as mean of "liked" movies from trg data
    ## and save it as pickle file
    ## --------------------------------------------------------------------
    user_vectors_dict = {}
    user_cnt = 0
    for _user_ID, _movie_IDs in dict_groups_trg.items():
        user_cnt += 1
        if user_cnt % 100 == 1:
            print("Calculating user vector for user {}".format(user_cnt))

        # user_movie_vectors = movie_vectors_df.loc[_movie_IDs] # pd.Series
        # user_movie_vectors = movie_vectors_df.reindex(_movie_IDs)
        # print(user_movie_vectors)
        # print()

        # user_vector = movie_vectors_df.loc[_movie_IDs].mean(axis=0).to_numpy()
        user_vector = movie_vectors_df.reindex(_movie_IDs).mean(axis=0).to_numpy()
        user_vectors_dict[_user_ID] = user_vector
        # print(user_vectors_dict)
        # break

    with open('Model/gensim/users/{}_user_vectors_dict.pickle'.format(MODEL_NAME), 'wb') as f:
        pickle.dump(user_vectors_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    ## --------------------------------------------------------------------


    ## --------------------------------------------------------------------
    ## Load validation data
    ## --------------------------------------------------------------------
    df_val = pd.read_hdf('Ratings/binarized.hdf', key='val')
    # print(df_val.head(5)) # 'movieID' is both 2nd level index and a column
    df_val = transform_df(df_val)
    # df_val = df_val.head(10) # quick check on 10 movies
    # print(df_val.head(5))
    # print()

    df_val_gb = df_val.groupby([USER_ID])
    # print(df_val_gb) # <pandas.core.groupby.generic.DataFrameGroupBy object>
    dict_groups_val = {k: list(v[MOVIE_ID]) for k, v in df_val_gb}
    # print(dict_groups_val[22085])
    # print()
    ## --------------------------------------------------------------------


    ## --------------------------------------------------------------------
    ## Retrieve user vectors from pickle file
    ## --------------------------------------------------------------------
    with open('Model/gensim/users/{}_user_vectors_dict.pickle'.format(MODEL_NAME), 'rb') as f:
        user_vectors_dict = pickle.load(f)

    ## Some simple testing on user 1 and rated terminator 1 and 2 movies
    '''
    user1_vec = user_vectors_dict[1].reshape(1,-1)
    terminator1_vec = movie_vectors_df.loc['589'].to_numpy().reshape(1, -1)
    terminator2_vec = movie_vectors_df.loc['1240'].to_numpy().reshape(1, -1)
    print(PAIRWISE_METRIC(terminator1_vec, terminator2_vec))
    print(PAIRWISE_METRIC(user1_vec, terminator1_vec))
    print(PAIRWISE_METRIC(user1_vec, terminator2_vec))
    print()
    # [[0.9180718]]
    # [[0.8272728]]
    # [[0.9124394]]

    starwars1_vec = movie_vectors_df.loc['2628'].to_numpy().reshape(1, -1)
    starwars2_vec = movie_vectors_df.loc['5378'].to_numpy().reshape(1, -1)
    starwars3_vec = movie_vectors_df.loc['33493'].to_numpy().reshape(1, -1)
    starwars4_vec = movie_vectors_df.loc['260'].to_numpy().reshape(1, -1)
    starwars5_vec = movie_vectors_df.loc['1196'].to_numpy().reshape(1, -1)
    starwars6_vec = movie_vectors_df.loc['1210'].to_numpy().reshape(1, -1)
    print(PAIRWISE_METRIC(starwars1_vec, starwars4_vec))
    print(PAIRWISE_METRIC(user1_vec, starwars1_vec))
    print(PAIRWISE_METRIC(user1_vec, starwars4_vec))
    # [[0.80287135]]
    # [[0.8336077]]
    # [[0.8185677]]
    print(PAIRWISE_METRIC(starwars1_vec, starwars2_vec))
    print(PAIRWISE_METRIC(starwars1_vec, starwars3_vec))
    print(PAIRWISE_METRIC(starwars4_vec, starwars5_vec))
    print(PAIRWISE_METRIC(starwars4_vec, starwars6_vec))
    # [[0.8580897]]
    # [[0.76029795]]
    # [[0.96574926]]
    # [[0.9652451]]
    '''

    ## Some simple testing of cosine similarity on user 22085
    # user_vector = user_vectors_dict[22085]
    # user_vector = user_vector.reshape(1,-1)  # sklearn cos_sim need 2D
    # print(user_vector)

    # # user_movie_IDs = dict_groups_trg[22085]
    # user_movie_IDs = dict_groups_val[22085][:10]
    # print(user_movie_IDs)

    # user_movie_vectors = movie_vectors_df.loc[user_movie_IDs].to_numpy()
    # print(user_movie_vectors)

    # cos_sim = PAIRWISE_METRIC(user_movie_vectors, user_vector)
    # print(cos_sim)
    # print(len(user_movie_IDs), len(user_movie_vectors), len(cos_sim))


    ## Alternative method using gensim's KeyVectors.distances
    ## vs sklearn.metrics.pairwise.cosine_similarity.
    ## Result is generally very close, with some rounding diffs
    # user_vector = user_vectors_dict[22085]
    # user_movie_IDs = dict_groups_val[22085]#[:10]
    # cos_sim = 1 - myModel.wv.distances(user_vector, user_movie_IDs)
    # print(cos_sim)
    ## --------------------------------------------------------------------


    ## --------------------------------------------------------------------
    ## Evaluate AUC of trained model on validation dataset
    ## --------------------------------------------------------------------

    ## quick testing to ensure user/movie IDs are in the same order
    ## between df_val and dict_groups_val
    '''
    # user_ratings = df_val.loc[22085]
    # print(user_ratings)
    # user_movie_IDs = dict_groups_val[22085]
    # print(user_movie_IDs)

    # print(df_val.index.get_level_values(USER_ID))
    # print(list(dict_groups_val.keys())[:10])
    # print(len(list(dict_groups_val.keys())))    # 138493 users
    '''

    if EVAL_DATASET == 'trg':
        df_eval = df_trg
        dict_groups_eval = dict_groups_trg
    else:
        df_eval = df_val
        dict_groups_eval = dict_groups_val

    scores = None
    user_cnt = 0
    for _user_ID, _movie_IDs in dict_groups_eval.items():
        user_cnt += 1
        if user_cnt % 100 == 1:
            print("Validating for user {}".format(user_cnt))

        user_vector = user_vectors_dict[_user_ID] # 1D

        if SCORE_METHOD == 0:
            user_vector = user_vector.reshape(1,-1) # sklearn cos_sim need 2D
        else:
            user_vector = user_vector.reshape(-1, 1) # for a.dot(b)

        ## reindex accounts for nan's; .loc won't handle nan in the futures
        # user_movie_vectors = movie_vectors_df.loc[_movie_IDs].to_numpy()
        user_movie_vectors = movie_vectors_df.reindex(_movie_IDs).to_numpy()

        ## Need to consider movies excluded due to word2vec minCount
        ## movie_vectors_df.loc[_movie_IDs] keep those as NaN
        ## Must exclude before comparing PAIRWISE_METRIC
        cos_sim = np.empty((len(user_movie_vectors), 1))
        cos_sim[:] = np.nan
        mask_not_nan = np.logical_not(np.isnan(user_movie_vectors))
        mask_not_nan = mask_not_nan[:, 0] # only need 1D mask

        if SCORE_METHOD == 0:
            cos_sim[mask_not_nan] = PAIRWISE_METRIC(
                                        user_movie_vectors[mask_not_nan],
                                        user_vector)
        else:
            cos_sim[mask_not_nan] = user_movie_vectors[mask_not_nan].dot(
                                    user_vector)

        if scores is None:
            scores = cos_sim.ravel() # flatten without making a copy
        else:
            scores = np.append(scores, cos_sim.ravel()) # flatten

    df_eval['scores'] = scores

    if SCORE_METHOD == 0:
        df_eval[[LIKED, 'scores']].to_csv('Model/gensim/eval/{}_{}_{}.csv'.format(
                                            MODEL_NAME, EVAL_DATASET, PAIRWISE_METRIC.__name__))
        with open('Model/gensim/eval/{}_{}_{}.pickle'.format(
                    MODEL_NAME, EVAL_DATASET, PAIRWISE_METRIC.__name__), 'wb') as f:
            pickle.dump(df_eval, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        df_eval[[LIKED, 'scores']].to_csv('Model/gensim/eval/{}_{}_dot.csv'.format(
                                            MODEL_NAME, EVAL_DATASET))
        with open('Model/gensim/eval/{}_{}_dot.pickle'.format(
                    MODEL_NAME, EVAL_DATASET), 'wb') as f:
            pickle.dump(df_eval, f, protocol=pickle.HIGHEST_PROTOCOL)

    mask_not_nan = np.logical_not(np.isnan(scores)) # ~ may also work
    truth = df_eval[LIKED].to_numpy()
    print(truth[mask_not_nan].min(), truth[mask_not_nan].max())
    print(scores[mask_not_nan].min(), scores[mask_not_nan].max())
    print(roc_auc_score(truth[mask_not_nan], scores[mask_not_nan]))
    ## --------------------------------------------------------------------



if __name__ == '__main__':
    start_time = datetime.now()

    run_validation_metrics()

    end_time = datetime.now()
    run_time = end_time - start_time
    print('Total Run Time: {}'.format(run_time))
