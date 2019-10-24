import numpy as np
import pandas as pd
from gensim.models import Word2Vec
# from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import roc_auc_score

import pickle
# import h5py
import os
import sys
import glob
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

MOVIE_PATH = 'ml-20m/movies.csv'
SAVED_MODEL_DIR = 'Model/gensim_models'
MODEL_PERF_DIR = 'Model/gensim_perf'


class MovieEdge(object):

    def __init__(self, movie_path, model_path, verbose=False):

        self.verbose = verbose

        ## load the movies data
        self.df_movies = pd.read_csv(movie_path, index_col=MOVIE_ID)
        if verbose:
            print('There are totally {} movies'.format(self.df_movies.index.size)) #

        ## load the Word2Vec model for pre-trained movie embeddings
        myModel = Word2Vec.load(model_path)
        self.movie2Vec = myModel.wv
        self.movie_IDs = list(self.movie2Vec.vocab.keys()) # trained movies
        if verbose:
            print('There are totally {} movie embeddings trained'.format(
                len(self.movie_IDs)))
            print()

        self.movie_vectors_dict = dict()
        for _movie_ID in self.movie_IDs:
            self.movie_vectors_dict[_movie_ID] = self.movie2Vec[_movie_ID]

        self.movie_vectors = np.array(list(self.movie_vectors_dict.values()))
        # self.movie_vectors_df = pd.DataFrame(self.movie_vectors, index=self.movie_IDs)

    def recommend(self, user_model, rated_movie_IDs=[], topn = 10):
        unrated_idx = [i for i, w in enumerate(self.movie_IDs) if w not in rated_movie_IDs]

        # unrated_movie_IDs = [self.movie_IDs[i] for i in unrated_idx]
        # unrated_movie_vectors = self.movie_vectors_df.loc[unrated_movie_IDs]

        ## same as 2 lines above without using pd.DataFrame
        unrated_movie_vectors = self.movie_vectors[unrated_idx]
        if self.verbose:
            print('Recommending top {} movies from {} unrated movies...'.format(
                    topn, len(unrated_idx)))
            print('excluding {} rated movies.'.format(len(rated_movie_IDs)))

        unrated_movie_scores = user_model.predict(unrated_movie_vectors).clip(0,1)
        if topn == -1:  # score all unrated movies
            unrated_topn_idx = unrated_movie_scores.argsort()[::-1]
        else:
            unrated_topn_idx = unrated_movie_scores.argsort()[::-1][:topn]

        unrated_movie_IDs = [self.movie_IDs[i] for i in unrated_idx]
        topn_movie_IDs = [unrated_movie_IDs[i] for i in unrated_topn_idx]

        ## quick test on user 1 matched previous validation results from AUROC run
        # test_movie_IDs = ['50', '112', '260', '589']
        # test_idx = [unrated_movie_IDs.index(w) for w in test_movie_IDs]
        # test_scores = unrated_movie_scores[test_idx]
        # print(list(zip(test_movie_IDs, list(test_scores))))

        topn_movie_IDs = [int(i) for i in topn_movie_IDs]
        df_pred = self.df_movies.loc[topn_movie_IDs]
        df_pred['score'] = unrated_movie_scores[unrated_topn_idx]

        return df_pred


if __name__ == '__main__':

    start_time = datetime.now()

    ## --------------------------------------------------------------------
    ## load movie data and pre-trained movie embeddings from Word2Vec model
    ## --------------------------------------------------------------------
    model_name = 'w2v_vs_16_sg_1_hs_1_mc_1_it_1_wn_32_ng_2.gensim'
    print('...loading movie data and pre-trained movie embeddings from')
    print('...Word2Vec model w2v_vs_16_sg_1_hs_1_mc_1_it_1_wn_32_ng_2.gensim')
    start_time1 = datetime.now()

    myMovieEdge = MovieEdge(MOVIE_PATH,
                            '{}/{}'.format(SAVED_MODEL_DIR, model_name),
                            verbose=True)

    end_time1 = datetime.now()
    run_time1 = end_time1 - start_time1
    print('~ Run Time: {}'.format(run_time1))
    print()
    ## --------------------------------------------------------------------

    ## --------------------------------------------------------------------
    ## load all users from pre-trained RidgeCV models
    ## --------------------------------------------------------------------
    ## Regularization of ridgeCV (linear least sq with L2) for SCORE_METHOD == 1
    alpha = [4.]
    print('...loading all users from pre-trained RidgeCV models')
    start_time1 = datetime.now()

    user_dir = '{}/users/ridge_a{}'.format(MODEL_PERF_DIR, alpha)
    with open('{}/{}_ridge_a{}_user_vectors_dict.pickle'.format(
                user_dir, model_name, alpha), 'rb') as f:
        user_vectors_dict = pickle.load(f)    

    end_time1 = datetime.now()
    run_time1 = end_time1 - start_time1
    print('~ Run Time: {}'.format(run_time1))
    print()

    ## --------------------------------------------------------------------
    ## Recommend topn movies for user 1
    ## --------------------------------------------------------------------
    print('...recommending topn movies for user 1')
    start_time1 = datetime.now()

    df_pred = myMovieEdge.recommend(user_vectors_dict[1], topn=10)

    end_time1 = datetime.now()
    run_time1 = end_time1 - start_time1

    print(df_pred)

    print('~ Run Time: {}'.format(run_time1))
    print()
    ## --------------------------------------------------------------------

    end_time = datetime.now()
    run_time = end_time - start_time
    print('~ Total Run Time: {}'.format(run_time))
