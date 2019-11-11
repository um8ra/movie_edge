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
        self.embed_movie_IDs = list(self.movie2Vec.vocab.keys()) # trained movies
        if verbose:
            print('There are totally {} movie embeddings trained'.format(
                len(self.embed_movie_IDs)))
            print()

        self.movie_vectors_dict = dict()
        for _movie_ID in self.embed_movie_IDs:
            self.movie_vectors_dict[_movie_ID] = self.movie2Vec[_movie_ID]

        self.movie_vectors = np.array(list(self.movie_vectors_dict.values()))
        self.movie_vectors_df = pd.DataFrame(self.movie_vectors, index=self.embed_movie_IDs)

    def update_user(self, liked_movie_IDs, disliked_movie_IDs, alpha=[8.]):
        movie_IDs = liked_movie_IDs + disliked_movie_IDs
        movie_IDs = [str(m) for m in movie_IDs]
        liked = np.zeros(len(movie_IDs), dtype=int)
        liked[:len(liked_movie_IDs)] = 1

        idx_in_vocab = [i for i, w in enumerate(movie_IDs) if w in self.embed_movie_IDs]
        movie_IDs_in_vocab = [movie_IDs[i] for i in idx_in_vocab]
        _movie_vectors = self.movie_vectors_df.loc[movie_IDs_in_vocab]
        _liked = liked[idx_in_vocab]

        print(movie_IDs)
        print(_movie_vectors)
        print(_liked)
        user_model = RidgeCV(alphas=alpha).fit(_movie_vectors, _liked)

        return user_model

    def recommend(self, user_model, liked_movie_IDs, disliked_movie_IDs, topn = 10):
        liked_movie_IDs = [str(m) for m in liked_movie_IDs]
        disliked_movie_IDs = [str(m) for m in disliked_movie_IDs]
        rated_movie_IDs = liked_movie_IDs + disliked_movie_IDs

        if self.verbose:
            print('Recommending top {} movies'.format(topn))
            print('excluding {} rated movies'.format(len(rated_movie_IDs)))

        if len(disliked_movie_IDs) == 0 and len(liked_movie_IDs) > 0:
            if self.verbose:
                print('Using word2vec most_similar')
                print('')

            aux = self.movie2Vec.most_similar(
                                        positive=liked_movie_IDs,
                                        negative=disliked_movie_IDs,
                                        topn=topn,
                                        indexer=None)
            topn_movie_IDs, topn_movie_scores = zip(*aux)
            topn_movie_IDs = [int(i) for i in topn_movie_IDs]
            print(topn_movie_IDs)
            print(topn_movie_scores)

            df_pred = self.df_movies.loc[topn_movie_IDs]
            df_pred['score'] = topn_movie_scores
        else:
            if self.verbose:
                print('Using loaded RidgeCV user model')
                print()

            unrated_idx = [i for i, w in enumerate(self.embed_movie_IDs) if w not in rated_movie_IDs]

            # unrated_movie_IDs = [self.embed_movie_IDs[i] for i in unrated_idx]
            # unrated_movie_vectors = self.movie_vectors_df.loc[unrated_movie_IDs]

            ## same as 2 lines above without using pd.DataFrame
            unrated_movie_vectors = self.movie_vectors[unrated_idx]
            unrated_movie_scores = user_model.predict(unrated_movie_vectors).clip(0,1)

            if topn == -1:  # score all unrated movies
                unrated_topn_idx = unrated_movie_scores.argsort()[::-1]
            else:
                unrated_topn_idx = unrated_movie_scores.argsort()[::-1][:topn]

            unrated_movie_IDs = [self.embed_movie_IDs[i] for i in unrated_idx]
            topn_movie_IDs = [unrated_movie_IDs[i] for i in unrated_topn_idx]

            ## quick test on user 1 matched previous validation results from AUROC run
            # testmovie_IDs = ['50', '112', '260', '589']
            # test_idx = [unrated_movie_IDs.index(w) for w in testmovie_IDs]
            # test_scores = unrated_movie_scores[test_idx]
            # print(list(zip(testmovie_IDs, list(test_scores))))

            print(self.movie_vectors_df.loc[topn_movie_IDs])
            topn_movie_IDs = [int(i) for i in topn_movie_IDs]
            df_pred = self.df_movies.loc[topn_movie_IDs]
            df_pred['score'] = unrated_movie_scores[unrated_topn_idx]

        return df_pred


if __name__ == '__main__':

    start_time = datetime.now()

    ## --------------------------------------------------------------------
    ## load movie data and pre-trained movie embeddings from Word2Vec model
    ## --------------------------------------------------------------------
    # model_name = 'w2v_vs_16_sg_1_hs_1_mc_1_it_1_wn_32_ng_2.gensim'
    model_name = 'w2v_vs_64_sg_1_hs_1_mc_1_it_4_wn_32_ng_2_all_data_trg_val.gensim'
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
    # alpha = [4.]
    alpha = [8.]
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

    df_pred = myMovieEdge.recommend(user_vectors_dict[1], [], [], topn=10)

    end_time1 = datetime.now()
    run_time1 = end_time1 - start_time1

    print(df_pred)

    print('~ Run Time: {}'.format(run_time1))
    print()
    ## --------------------------------------------------------------------

    end_time = datetime.now()
    run_time = end_time - start_time
    print('~ Total Run Time: {}'.format(run_time))


    ## --------------------------------------------------------------------
    ## Train a new user RidgeCV model from liked and disliked movies
    ## --------------------------------------------------------------------
    print('...recommending topn movies for new user')
    start_time1 = datetime.now()

    liked = [2571,  # Matrix 1
             6365,  # Matrix 2
             6934,  # Matrix 3
             589]   # Terminator 2
    disliked = [78245,  # Romeo and Juliet
                125916, # Fifty Shades of Grey
                26505,  # Comfort and Joy
                59846]  # The Iron Mask (1929)
    disliked = [125916]

    user_model = myMovieEdge.update_user(liked_movie_IDs=liked,
                                         disliked_movie_IDs=disliked)
    df_pred = myMovieEdge.recommend(user_model,
                                    liked_movie_IDs=liked,
                                    disliked_movie_IDs=disliked,
                                    topn=10)

    end_time1 = datetime.now()
    run_time1 = end_time1 - start_time1

    print(df_pred)

    print('~ Run Time: {}'.format(run_time1))
    print()
