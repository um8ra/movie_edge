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

SAVED_MODEL_DIR = 'Model/gensim_models'
MODEL_PERF_DIR = 'Model/gensim_perf'
OVERRIDE = False


# alphas = [[100, 10., 5., 1., 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]]
# alphas.reverse()

PAIRWISE_METRIC = cosine_similarity # if score_method == 0
# PAIRWISE_METRIC = euclidean_distances  # if score_method == 0




def run_validation_metrics(eval_dataset, model_name, score_method=1, alpha=1e-2):

    loaded_trg_all = False

    print('Evaluating on {} set'.format(eval_dataset))
    print('Word2Vec model: ', model_name)
    print('score method :', score_method)
    print('alpha: ', alpha)
    print()

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

    myModel = Word2Vec.load('{}/{}'.format(
                SAVED_MODEL_DIR, model_name))
    movie2vec_kv = myModel.wv
    # print(type(movie2vec_kv))     # <class 'gensim.models.keyedvectors.Word2VecKeyedVectors'>
    # print(type(myModel.wv.vocab)) # dict, values are gensim.models.keyedvectors.Vocab object
    # print(list(myModel.wv.vocab.keys())[0:10])    
    # print(myModel.wv.vocab['589'])
    # print()
    # print(type(myModel.wv.index2word)) # list
    # print(myModel.wv.index2word[0:10])

    vocab = list(myModel.wv.vocab.keys())

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

    ## same result for 4 options below
    # print(myModel.wv['589'])
    # print(myModel.wv.get_vector('589'))
    # print(myModel.wv.word_vec('589'))
    # print(movie2vec_dict['589'])

    # # test_movie_vectors = movie2vec(df_movies, myModel, 'Saving Private Ryan') # '2028'
    # test_movie_vectors = movie2vec(df_movies, myModel, 'Terminator 2: Judgment Day') # '589'
    # print(test_movie_vectors)
    # print()


    '''
    ## Testing gensim's most_similar(pos, neg) instead of user vector method
    vocab = list(myModel.wv.vocab.keys())
    # print(vocab)    # it's NOT sorted
    positive = dict_groups_trg[1]
    negative =  dict_groups_trg0[1]
    ## filter out movies that are not in the trained vocab
    idx_in_vocab = [i for i, w in enumerate(negative) if w in vocab]
    negative = [negative[i] for i in idx_in_vocab]
    _score = myModel.wv.most_similar(positive=positive, # user1 LIKED
                                      negative=negative, # user1 not LIKED
                                      # topn=5, # returns (movie_ID, score)
                                      # topn=len(vocab),  # similarity for all words
                                      topn=None, # similarity for all words
                                      restrict_vocab=None,
                                      indexer=None)
    # _score = myModel.wv.most_similar_cosmul(positive=positive, # user1 LIKED
    #                                          negative=negative, # user1 not LIKED
    #                                          topn=None)  # similarity for all words

    print(len(_score))
    print(_score[:10])

    _score_dict = {k: v for k, v in zip(vocab, _score)}
    print(_score_dict['589'])
    print(_score_dict['1240'])
    print(_score_dict['2571'])
    print(_score_dict['6365'])
    print(_score_dict['6934'])
    print(_score_dict['2628'])
    print(_score_dict['5378'])
    print(_score_dict['33493'])
    print(_score_dict['260'])
    print(_score_dict['1196'])
    print(_score_dict['1210'])
    print()
    val_movie_IDs = ['50', '112', '260', '589', '1136', '1198', '1200', '1208',
                     '1222', '1240', '1278', '1350', '1358', '1370', '1848',
                     '1997', '2194', '2288', '2761', '2804', '3037', '3479',
                     '4027', '5040', '7454', '8507']
    for _movie_ID in val_movie_IDs:
        print(_movie_ID, _score_dict[_movie_ID])
    print()
    '''

    ## Test vs Jonathan's MF test cases
    '''
    print('>> Star Wars IV (260) vs other Star Wars')
    starwars1_vec = movie_vectors_df.loc['2628'].to_numpy().reshape(1, -1)
    starwars2_vec = movie_vectors_df.loc['5378'].to_numpy().reshape(1, -1)
    starwars3_vec = movie_vectors_df.loc['33493'].to_numpy().reshape(1, -1)
    starwars4_vec = movie_vectors_df.loc['260'].to_numpy().reshape(1, -1)
    starwars5_vec = movie_vectors_df.loc['1196'].to_numpy().reshape(1, -1)
    starwars6_vec = movie_vectors_df.loc['1210'].to_numpy().reshape(1, -1)
    print(PAIRWISE_METRIC(starwars4_vec, starwars5_vec))
    print(PAIRWISE_METRIC(starwars4_vec, starwars6_vec))
    print(PAIRWISE_METRIC(starwars4_vec, starwars1_vec))
    print(PAIRWISE_METRIC(starwars4_vec, starwars2_vec))
    print(PAIRWISE_METRIC(starwars4_vec, starwars3_vec))
    print('>> Top 10 matches')
    # print(myModel.wv.similar_by_word('260'), topn=5)
    # print(myModel.wv.most_similar('260', topn=5))
    pred = myModel.wv.most_similar('260', topn=10)
    pred_movie_IDs, pred_score = zip(*pred)
    pred_movie_IDs = [int(i) for i in pred_movie_IDs]
    print(pred_movie_IDs)
    df_pred = df_movies.loc[pred_movie_IDs]
    df_pred['score'] = pred_score
    print(df_pred)
    print()

    print('>> Matrix (2571) vs other Matrix movies')
    matrix1_vec = movie_vectors_df.loc['2571'].to_numpy().reshape(1, -1)
    matrix2_vec = movie_vectors_df.loc['6365'].to_numpy().reshape(1, -1)
    matrix3_vec = movie_vectors_df.loc['6934'].to_numpy().reshape(1, -1)
    matrix_anime_vec = movie_vectors_df.loc['27660'].to_numpy().reshape(1, -1)
    print(PAIRWISE_METRIC(matrix1_vec, matrix2_vec))
    print(PAIRWISE_METRIC(matrix1_vec, matrix3_vec))
    print(PAIRWISE_METRIC(matrix1_vec, matrix_anime_vec))
    print('>> Top 10 matches')
    # print(myModel.wv.similar_by_word('2571', topn=10))
    # print(myModel.wv.most_similar('2571', topn=10))
    pred = myModel.wv.most_similar('2571', topn=10)
    pred_movie_IDs, pred_score = zip(*pred)
    pred_movie_IDs = [int(i) for i in pred_movie_IDs]
    print(pred_movie_IDs)
    df_pred = df_movies.loc[pred_movie_IDs]
    df_pred['score'] = pred_score
    print(df_pred)
    print()

    # print('>> Carnages (6622)')
    print('>> Crouching Tiger, Hidden Dragon (3996)')
    print('>> Top 10 matches')
    # print(myModel.wv.similar_by_word('2571', topn=10))
    # print(myModel.wv.most_similar('2571', topn=10))
    pred = myModel.wv.most_similar('3996', topn=10)
    pred_movie_IDs, pred_score = zip(*pred)
    pred_movie_IDs = [int(i) for i in pred_movie_IDs]
    print(pred_movie_IDs)
    df_pred = df_movies.loc[pred_movie_IDs]
    df_pred['score'] = pred_score
    print(df_pred)
    print()
    '''
    ## --------------------------------------------------------------


    ## --------------------------------------------------------------------
    ## Load training data
    ## --------------------------------------------------------------------
    df_trg_all = pd.read_hdf('Ratings/binarized.hdf', key='trg')
    df_trg = df_trg_all[df_trg_all[LIKED] == 1]
    df_trg = transform_df(df_trg)
    # df_trg = df_trg.head(200) # quick check on 10 movies

    df_trg_gb = df_trg.groupby([USER_ID])
    dict_groups_trg = {k: list(v[MOVIE_ID]) for k, v in df_trg_gb}

    if score_method == 1:
        ## To train a RidgeCV model for each user

        loaded_trg_all = True

        df_trg_all = transform_df(df_trg_all)
        # df_trg_all = df_trg_all.head(200) # quick check on 10 movies

        if eval_dataset in ['trg+val', 'tst']:
            ## RidgeCV to train user on trg+val set
            ## then later evaluate on trg+val or tst set
            df_val = pd.read_hdf('Ratings/binarized.hdf', key='val')
            df_val = transform_df(df_val)
            # df_val = df_val.head(50) # quick check on 10 movies
            df_trg_all = pd.concat([df_trg_all, df_val])
            ## the same user is in 2 different parts of df after concat
            ## must sort so merged user-movie predicitons from each user
            ## in dict_groups_trg_eval are in the same order of df_trg_all
            df_trg_all.sort_index(inplace=True)

        df_trg_all_gb = df_trg_all.groupby([USER_ID])
        dict_groups_trg_all = {k: {MOVIE_ID: list(v[MOVIE_ID]),
                                   LIKED: v[LIKED].to_numpy()}
                                for k, v in df_trg_all_gb}

    if score_method == 2:
        df_trg0 = df_trg_all[df_trg_all[LIKED] == 0]
        df_trg0 = transform_df(df_trg0)
        # df_trg0 = df_trg0.head(200) # quick check on 10 movies
        df_trg0_gb = df_trg0.groupby([USER_ID])
        dict_groups_trg0 = {k: list(v[MOVIE_ID]) for k, v in df_trg0_gb}
    ## --------------------------------------------------------------------


    ## ********************************************************************
    ## skip this steps if user_vectors_dict is saved to pickle
    ## --------------------------------------------------------------------
    ## Calculate each user vector as mean of "liked" movies from trg data
    ## and save it as pickle file
    ## --------------------------------------------------------------------
    if score_method == 1:
        output_dir = '{}/users/ridge_a{}'.format(MODEL_PERF_DIR, alpha)
        output_fname = '{}/{}_ridge_a{}_user_vectors_dict.pickle'.format(
                            output_dir, model_name, alpha)

        if os.path.exists(output_fname) and OVERRIDE == False:
            with open(output_fname, 'rb') as f:
                user_vectors_dict = pickle.load(f)
        else:
            user_vectors_dict = {}
            user_cnt = 0
            for _user_ID, _value in dict_groups_trg_all.items():
                _movie_IDs = _value[MOVIE_ID]
                _liked = _value[LIKED]

                user_cnt += 1
                if user_cnt % 100 == 1:
                    print("Calculating user vector for user {}".format(user_cnt))

                idx_in_vocab = [i for i, w in enumerate(_movie_IDs) if w in vocab]
                _movie_IDs_in_vocab = [_movie_IDs[i] for i in idx_in_vocab]
                _movie_vectors = movie_vectors_df.loc[_movie_IDs_in_vocab]
                _movie_liked = _liked[idx_in_vocab]

                clf = RidgeCV(alphas=alpha).fit(_movie_vectors, _movie_liked)
                user_vectors_dict[_user_ID] = clf

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            with open(output_fname, 'wb') as f:
                pickle.dump(user_vectors_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif score_method == 2:
        pass

    else:
        output_dir = '{}/users/mean_movies'.format(MODEL_PERF_DIR)
        output_fname = '{}/{}_mean_movies_user_vectors_dict.pickle'.format(
                            output_dir, model_name)

        if os.path.exists(output_fname) and OVERRIDE == False:
            with open(output_fname, 'rb') as f:
                user_vectors_dict = pickle.load(f)
        else:
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

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            with open(output_fname, 'wb') as f:
                pickle.dump(user_vectors_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    ## --------------------------------------------------------------------


    ## --------------------------------------------------------------------
    ## Load validation data
    ## --------------------------------------------------------------------
    if eval_dataset in ['trg', 'trg+val']:
        ## partially prepared during training set prep earlier
        ## note that dict_groups_trg_eval has a different dict structure
        ## compared to dict_groups_trg_all. The latter contain sub key-value
        ## 'movieID' and 'Liked' ratings to train RidgeCV. The former just
        ## have all the movie_IDs per user_ID key value for evaluation.
        ## Predicted validation scores are appended then compared to df_trg_all. 
        if not loaded_trg_all:
            df_trg_all = pd.read_hdf('Ratings/binarized.hdf', key='trg')
            df_trg_all = transform_df(df_trg_all)
            # df_trg_all = df_trg_all.head(200) # quick check on 10 movies

            if eval_dataset == 'trg+val':
                df_val = pd.read_hdf('Ratings/binarized.hdf', key='val')
                df_val = transform_df(df_val)
                # df_val = df_val.head(50) # quick check on 10 movies
                df_trg_all = pd.concat([df_trg_all, df_val])
                ## the same user is in 2 different parts of df after concat
                ## must sort so merged user-movie predicitons from each user
                ## in dict_groups_trg_eval are in the same order of df_trg_all
                df_trg_all.sort_index(inplace=True)

            df_trg_all_gb = df_trg_all.groupby([USER_ID])
        dict_groups_trg_eval = {k: list(v[MOVIE_ID]) for k, v in df_trg_all_gb}

    elif eval_dataset == 'val':
        df_val = pd.read_hdf('Ratings/binarized.hdf', key=eval_dataset)
        # print(df_val.head(5)) # 'movieID' is both 2nd level index and a column
        df_val = transform_df(df_val)
        # df_val = df_val.head(50) # quick check on 10 movies
        # print(df_val.head(5))
        # print()

        df_val_gb = df_val.groupby([USER_ID])
        # print(df_val_gb) # <pandas.core.groupby.generic.DataFrameGroupBy object>
        dict_groups_val = {k: list(v[MOVIE_ID]) for k, v in df_val_gb}
        # print(dict_groups_val[22085])
        # print()

    elif eval_dataset == 'tst':
        df_tst = pd.read_hdf('Ratings/binarized.hdf', key=eval_dataset)
        # print(df_tst.head(5)) # 'movieID' is both 2nd level index and a column
        df_tst = transform_df(df_tst)
        # df_tst = df_tst.head(50) # quick check on 10 movies
        # print(df_tst.head(5))
        # print()

        df_tst_gb = df_tst.groupby([USER_ID])
        # print(df_tst_gb) # <pandas.core.groupby.generic.DataFrameGroupBy object>
        dict_groups_tst = {k: list(v[MOVIE_ID]) for k, v in df_tst_gb}
        # print(dict_groups_val[22085])
        # print()

    else:
        ## should not happen
        pass

    ## --------------------------------------------------------------------


    ## --------------------------------------------------------------------
    ## Retrieve user vectors from pickle file
    ## These are moved into the "calculating user vector" steps
    ## A file check is done; load existing pickle if file exists
    ## --------------------------------------------------------------------
    '''
    if score_method == 1:
        output_dir = '{}/users/ridge_a{}'.format(MODEL_PERF_DIR, alpha)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        with open('{}/{}_ridge_a{}_user_vectors_dict.pickle'.format(
                    output_dir, model_name, alpha), 'rb') as f:
            user_vectors_dict = pickle.load(f)    
    elif score_method == 2:
        pass
    else:
        output_dir = '{}/users/mean_movies'.format(MODEL_PERF_DIR)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        with open('{}/{}_mean_movies_user_vectors_dict.pickle'.format(
                    output_dir, model_name), 'rb') as f:
            user_vectors_dict = pickle.load(f)
    '''

    ## Some simple testing on user 1 and terminator / star wars
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

    matrix1_vec = movie_vectors_df.loc['2571'].to_numpy().reshape(1, -1)
    matrix2_vec = movie_vectors_df.loc['6365'].to_numpy().reshape(1, -1)
    matrix3_vec = movie_vectors_df.loc['6934'].to_numpy().reshape(1, -1)
    print(PAIRWISE_METRIC(matrix1_vec, matrix2_vec))
    print(PAIRWISE_METRIC(matrix1_vec, matrix3_vec))
    print()

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

    ## Some simple testing of cosine similarity on user 22085
    # user_vector = user_vectors_dict[22085]
    # user_vector = user_vector.reshape(1,-1)  # sklearn cos sim need 2D
    # print(user_vector)

    # # user_movie_IDs = dict_groups_trg[22085]
    # user_movie_IDs = dict_groups_val[22085][:10]
    # print(user_movie_IDs)

    # user_movie_vectors = movie_vectors_df.loc[user_movie_IDs].to_numpy()
    # print(user_movie_vectors)

    # _score = PAIRWISE_METRIC(user_movie_vectors, user_vector)
    # print(_score)
    # print(len(user_movie_IDs), len(user_movie_vectors), len(_score))


    ## Alternative method using gensim's KeyVectors.distances
    ## vs sklearn.metrics.pairwise.cosine_similarity.
    ## Result is generally very close, with some rounding diffs
    # user_vector = user_vectors_dict[22085]
    # user_movie_IDs = dict_groups_val[22085]#[:10]
    # _score = 1 - myModel.wv.distances(user_vector, user_movie_IDs)
    # print(_score)
    '''
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

    if eval_dataset in ['trg', 'trg+val']:
        df_eval = df_trg_all
        dict_groups_eval = dict_groups_trg_eval
    elif eval_dataset == 'val':
        df_eval = df_val
        dict_groups_eval = dict_groups_val
    elif eval_dataset == 'tst':
        df_eval = df_tst
        dict_groups_eval = dict_groups_tst
    else:
        pass

    scores = None
    user_cnt = 0
    for _user_ID, _movie_IDs in dict_groups_eval.items():
        user_cnt += 1
        if user_cnt % 100 == 1:
            print("Validating for user {}".format(user_cnt))

        ## load objects related to users
        if score_method == 0:
            user_vector = user_vectors_dict[_user_ID] # 1D
            user_vector = user_vector.reshape(1,-1) # sklearn cos sim need 2D
        elif score_method == 1:
            clf = user_vectors_dict[_user_ID]
        elif score_method == 2:
            positive = dict_groups_trg[1]
            negative =  dict_groups_trg0[1]
            ## filter out movies that are not in the trained vocab
            idx_in_vocab = [i for i, w in enumerate(negative) if w in vocab]
            negative = [negative[i] for i in idx_in_vocab]
        else:
            user_vector = user_vectors_dict[_user_ID] # 1D
            user_vector = user_vector.reshape(-1, 1) # transpose b for a.dot(b)

        ## Need to consider movies excluded due to word2vec minCount
        ## or movies in validation set not trained in training set
        ## Must exclude before comparing with similarity metrics
        # _score = np.empty((len(user_movie_vectors), 1))
        _score = np.empty((len(_movie_IDs), 1))
        _score[:] = np.nan

        if score_method not in (1, 2):
            ## reindex accounts for nan's; .loc won't handle nan in the futures
            # user_movie_vectors = movie_vectors_df.loc[_movie_IDs].to_numpy()
            user_movie_vectors = movie_vectors_df.reindex(_movie_IDs).to_numpy()

            mask_not_nan = np.logical_not(np.isnan(user_movie_vectors))
            mask_not_nan = mask_not_nan[:, 0] # only need 1D mask

        if score_method == 0:
            _score[mask_not_nan] = PAIRWISE_METRIC(
                                        user_movie_vectors[mask_not_nan],
                                        user_vector)
        elif score_method == 1:
            ## filter out movies that are not in the trained vocab
            idx_in_vocab = [i for i, w in enumerate(_movie_IDs) if w in vocab]
            _movie_IDs_in_vocab = [_movie_IDs[i] for i in idx_in_vocab]
            _score[idx_in_vocab, 0] = \
                 clf.predict(movie_vectors_df.loc[_movie_IDs_in_vocab].to_numpy()).clip(0,1)
        elif score_method == 2:
            _score_aux = myModel.wv.most_similar(
                                    positive=positive, # user1 LIKED
                                    negative=negative, # user1 not LIKED
                                    # topn=5, # returns (movie_ID, score)
                                    # topn=len(vocab),  # similarity for all words
                                    topn=None, # similarity for all words
                                    restrict_vocab=None,
                                    indexer=None)
            ## when topn=None above, only a 1D array is returned without index
            _score_dict = {k: v for k, v in zip(vocab, _score_aux)}
            idx_in_vocab = [i for i, w in enumerate(_movie_IDs) if w in vocab]
            _movie_IDs_in_vocab = [_movie_IDs[i] for i in idx_in_vocab]
            _score_aux = [_score_dict[i] for i in _movie_IDs_in_vocab]
            # _score_aux = [_score_dict[_movie_IDs[i]] for i in idx_in_vocab]
            _score[idx_in_vocab, 0] = _score_aux
        else:
            _score[mask_not_nan] = user_movie_vectors[mask_not_nan].dot(
                                    user_vector)

        if scores is None:
            scores = _score.ravel() # flatten without making a copy
        else:
            scores = np.append(scores, _score.ravel()) # flatten

    df_eval['scores'] = scores

    if score_method == 0:
        subdir = 'mean_movies_' + PAIRWISE_METRIC.__name__
    elif score_method == 1:
        subdir = 'ridge_a{}'.format(alpha)
    elif score_method == 2:
        subdir = 'most_similar'
    else:
        subdir = 'dot'

    output_dir = '{}/eval/{}/{}'.format(MODEL_PERF_DIR, eval_dataset, subdir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    df_eval[[LIKED, 'scores']].to_csv('{}/{}_{}_{}.csv'.format(
                output_dir, model_name, eval_dataset, subdir))
    with open('{}/{}_{}_{}.pickle'.format(
                output_dir, model_name, eval_dataset, subdir), 'wb') as f:
        pickle.dump(df_eval, f, protocol=pickle.HIGHEST_PROTOCOL)

    mask_not_nan = np.logical_not(np.isnan(scores)) # ~ may also work
    truth = df_eval[LIKED].to_numpy()   # LIKED is the column 'Liked' (1 or 0)
    truth_min = truth[mask_not_nan].min()
    truth_max = truth[mask_not_nan].max()
    score_min = scores[mask_not_nan].min()
    score_max = scores[mask_not_nan].max()
    auc = roc_auc_score(truth[mask_not_nan], scores[mask_not_nan])
    
    print(truth_min, truth_max)
    print(score_min, score_max)
    print(auc)

    ## Write some quick summaries into a txt file for each model
    with open('{}/{}_{}_{}_AUC.txt'.format(
                output_dir, model_name, eval_dataset, subdir), 'w') as f:
        f.write('Word2Vec model: {}\n'.format(model_name))
        f.write('Truth min: {}, max: {}\n'.format(truth_min, truth_max))
        f.write('Scores min: {}, max: {}\n'.format(score_min, score_max))
        f.write('AUC: {}\n'.format(auc))

    ## Append each model's AUC summary to a line in a csv file
    output_fname = '{}/eval/{}/{}_AUC_comparisons.csv'.format(
                        MODEL_PERF_DIR, eval_dataset, eval_dataset)
    print(output_fname)
    if os.path.exists(output_fname):
        open_mode = 'a' # append if already exists
    else:
        open_mode = 'w' # make a new file if not
    with open(output_fname, open_mode) as f:
        if score_method == 0:
            f.write(model_name + ',' + 'mean_user' + ',' + str(auc) + '\n')
        elif score_method == 1:
            f.write(model_name + ',' + str(alpha) + ',' + str(auc) + '\n')
        elif score_method == 2:
            f.write(model_name + ',' + 'most_similar' + ',' + str(auc) + '\n')
        else:
            f.write(model_name + ',' + 'dot_product' + ',' + str(auc) + '\n')
    ## --------------------------------------------------------------------



if __name__ == '__main__':
    start_time = datetime.now()

    ## Note that dot product may not actually make sense
    ## cosine similarity is between -1 and 1, with 1 being more similar.
    ## If we do dot product, it would have norm(vec1) and norm(vec2)
    ## as magnitute multiplied on top of the cos sim.
    ## The euclidean distance would be further with large norms.
    # score_method = 0 # 0: user vector = mean of movies; sklearn pairwise metric
    #                  # 1: user vector = ridgecv; sklearn pairwise metric
    #                  # 2: gensim myModel.vw.most_similar(pos,neg)
    #                  # else: dot product
    score_methods = [1, 0]

    ## Regularization of ridgeCV (linear least sq with L2) for score_method == 1
    # alphas = [[10.]]
    # alphas = [[100.], [10.], [1.], [1e-1], [1e-2], [1e-3], [1e-4], [1e-5]]
    # alphas = [[50.], [40.], [30.], [20.]]
    # alphas = [[5.], [15.]]
    # alphas = [[4.], [6.]]
    # alphas = [[2.], [3.]]
    # alphas = [[4.]]
    # alphas = [[10.], [4.], [1.], [1e-1], [1e-2], [1e-3], [1e-4], [1e-5]]
    # alphas = [[100.], [20.], [1.], [1e-1], [1e-2], [1e-3], [1e-4], [1e-5]]
    # alphas = [[10.], [8.], [6.], [4.], [2.], [1.]]
    alphas = [[8.]]

    # eval_dataset = 'trg' # 'trg' or 'val'
    # eval_dataset = 'val' # 'trg' or 'val'
    # eval_dataset = 'tst' # 'trg' or 'val'
    # eval_dataset = 'trg+val' # 'trg' or 'val'
    eval_datasets = ['trg+val', 'tst']
    # eval_datasets = ['trg+val']

    # model_names = []
    # for root, dirs, files in os.walk(SAVED_MODEL_DIR):
    #     for file in files:
    #         # exclude 'vs_128' and 'hs_0' for now
    #         # if '.gensim' in file and 'vs_128' not in file \
    #         if '.gensim' in file and 'vs_128' in file \
    #         and 'hs_0' not in file:
    #             model_names.append(file)

    # model_names = ['w2v_vs_16_sg_1_hs_1_mc_1_it_1_wn_32_ng_2.gensim']
    # model_names = ['w2v_vs_64_sg_1_hs_1_mc_1_it_1_wn_32_ng_2.gensim']

    # model_names = ['w2v_vs_64_sg_1_hs_1_mc_1_it_4_wn_32_ng_2.gensim']
    model_names = ['w2v_vs_64_sg_1_hs_1_mc_1_it_4_wn_32_ng_2_all_data_trg_val.gensim']
    # model_names = ['w2v_vs_64_sg_1_hs_1_mc_1_it_1_wn_32_ng_2.gensim',
    #                'w2v_vs_64_sg_1_hs_1_mc_1_it_2_wn_32_ng_2.gensim',
    #                'w2v_vs_64_sg_1_hs_1_mc_1_it_4_wn_32_ng_2.gensim']

    print('Number of models to evaluate: ', len(model_names))   # 83 models excluding '128'
    for m in model_names:
        print(m)
    print()

    for score_method in score_methods:
        for eval_dataset in eval_datasets:
            for model_name in model_names:
                for alpha in alphas:

                    start_time1 = datetime.now()

                    run_validation_metrics(eval_dataset, model_name, score_method, alpha)

                    end_time1 = datetime.now()
                    run_time1 = end_time1 - start_time1
                    print('1x Model Run Time: {}'.format(run_time1))

                    if score_method == 0:
                        subdir = 'mean_movies_' + PAIRWISE_METRIC.__name__
                    elif score_method == 1:
                        subdir = 'ridge_a{}'.format(alpha)
                    elif score_method == 2:
                        subdir = 'most_similar'
                    else:
                        subdir = 'dot'

                    ## Append run time stats to a txt file for each model
                    output_dir = '{}/eval/{}/{}'.format(MODEL_PERF_DIR, eval_dataset, subdir)
                    with open('{}/{}_{}_{}_AUC.txt'.format(
                            output_dir, model_name, eval_dataset, subdir), 'a') as f:
                        f.write('Run time: {}\n'.format(run_time1))

                    print()

    end_time = datetime.now()
    run_time = end_time - start_time
    print('Total Run Time: {}'.format(run_time))
