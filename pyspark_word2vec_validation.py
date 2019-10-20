import numpy as np
import pandas as pd
from pyspark.mllib.feature import Word2Vec, Word2VecModel
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext, Row
# from pyspark.ml.classification import LogisticRegression
# from pyspark.mllib.linalg import DenseVector
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

## if evaluating AUROC on trg, COMMENT out df_trg = df_trg[df_trg[LIKED] == 1]
EVAL_DATASET = 'val' # 'trg' or 'val'


# print(os.environ.get("SPARK_HOME"))


def run_validation_metrics():

    # partitions = os.cpu_count() - 2
    # sc = SparkContext('local[{cpus}]'.format(cpus=partitions), 'word2vec')

    # sc = SparkContext('local', 'word2vec')

    conf = SparkConf().setAppName('word2vec').setMaster('local')
    sc = SparkContext(conf=conf)
    # spark = SparkSession(sc)    # required for pyspark.rdd.RDD.toDF()
    sqlContext = SQLContext(sc) # required for pyspark.rdd.RDD.toDF() and read.parquet()
    # print(type(sc))         # <class 'pyspark.context.SparkContext'>
    # print(type(spark))      # <class 'pyspark.sql.session.SparkSession'>
    # print(type(sqlContext)) # <class 'pyspark.sql.context.SQLContext'>


    ## --------------------------------------------------------------
    ## Used helper functions for validation purpose
    ## --------------------------------------------------------------
    def transform_df(_df):
        _df = _df.drop([TIMESTAMP], axis=1)
        _df[MOVIE_ID] = _df.index.get_level_values(MOVIE_ID).astype(str)
        return _df

    df_movies = pd.read_csv('ml-20m/movies.csv', index_col=MOVIE_ID)
    print('There are totally {} movies'.format(df_movies.index.size)) # 27278
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

    def movie2vec(df_movies, model, search_str):
        movie_list = list()
        movie_index = df_movies[df_movies[TITLE].str.match(search_str)]
        print(movie_index)
        for mi in movie_index.index:
            movie_list.extend([(mi, model.transform(str(mi)))])
        return movie_list
    ## --------------------------------------------------------------


    ## --------------------------------------------------------------
    ## Load pre-trained Word2Vec Model into JavaMap dict-like object
    ## --------------------------------------------------------------
    '''
    myModel = Word2VecModel.load(sc, 'Model/trained_wor2vec_pyspark.sparkmodel')
    movie2vec_map = myModel.getVectors()
    # print(type(movie2vec_map))    # <class 'py4j.java_collections.JavaMap'>
    print('There are totally {} movie embeddings'.format(len(movie2vec_map))) # 16066
    ## getMinCount is only available from ml library, not mllib version
    # print('Movie minCount was set to {}'.format(myModel.getMinCount()))
    print()

    # test_movie_vectors = movie2vec(df_movies, myModel, 'Saving Private Ryan')
    # print(test_movie_vectors)
    # print()

    # print(movie2vec_map['2028'])    # 'Saving Private Ryan'
    # print()
    '''
    ## --------------------------------------------------------------


    ## --------------------------------------------------------------------
    ## Load pre-trained model to a Python dict, then key and df values
    ## --------------------------------------------------------------------
    movie2vec_df = sqlContext.read.parquet('Model/trained_wor2vec_pyspark.sparkmodel/data') \
                                  .alias("movie2vec_df")
    # print(type(movie2vec_df))     # <class 'pyspark.sql.dataframe.DataFrame'>
    # movie2vec_df.printSchema()
    movie2vec_dict = movie2vec_df.rdd.collectAsMap()
    print('There are totally {} movie embeddings'.format(len(movie2vec_dict))) # 16066
    # print(type(movie2vec_dict))   # dict

    # print(movie2vec_dict['2028']) # 'Saving Private Ryan'
    # print()

    movie_IDs = list(movie2vec_dict.keys())
    movie_vectors = np.array(list(movie2vec_dict.values()))
    movie_vectors_df = pd.DataFrame(movie_vectors, index=movie_IDs)
    # print(movie_vectors_df.loc[['2028', '10']])   # two movies
    # print()
    ## --------------------------------------------------------------------


    ## ********************************************************************
    ## skip the next 2 steps if user_vectors_dict is saved to pickle
    ##
    ## --------------------------------------------------------------------
    ## Load training data
    ## --------------------------------------------------------------------
    '''
    df_trg = pd.read_hdf('Ratings/binarized.hdf', key='trg')
    df_trg = df_trg[df_trg[LIKED] == 1] # comment out if evaluating AUROC
    # df_trg = df_trg.head(50000) # todo comment out for production
    df_trg = transform_df(df_trg)
    # df_trg = df_trg.head(10) # quick check on 10 movies

    df_trg_gb = df_trg.groupby([USER_ID])
    dict_groups_trg = {k: list(v[MOVIE_ID]) for k, v in df_trg_gb}

    # document = sc.parallelize(dict_groups_trg.values(), partitions)
    '''
    ## --------------------------------------------------------------------


    ## --------------------------------------------------------------------
    ## Calculate each user vector as mean of "liked" movies from trg data
    ## and save it as pickle file
    ## --------------------------------------------------------------------
    ''' 
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

    with open('Model/word2vec_pyspark_0_user_vectors_dict.pickle', 'wb') as f:
        pickle.dump(user_vectors_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    '''
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
    with open('Model/word2vec_pyspark_0_user_vectors_dict.pickle', 'rb') as f:
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
    # [[0.90550922]]
    # [[0.20541165]]
    # [[0.46076321]]

    starwars1_vec = movie_vectors_df.loc['2628'].to_numpy().reshape(1, -1)
    starwars2_vec = movie_vectors_df.loc['5378'].to_numpy().reshape(1, -1)
    starwars3_vec = movie_vectors_df.loc['33493'].to_numpy().reshape(1, -1)
    starwars4_vec = movie_vectors_df.loc['260'].to_numpy().reshape(1, -1)
    starwars5_vec = movie_vectors_df.loc['1196'].to_numpy().reshape(1, -1)
    starwars6_vec = movie_vectors_df.loc['1210'].to_numpy().reshape(1, -1)
    print(PAIRWISE_METRIC(starwars1_vec, starwars4_vec))
    print(PAIRWISE_METRIC(user1_vec, starwars1_vec))
    print(PAIRWISE_METRIC(user1_vec, starwars4_vec))
    # [[0.14618965]]
    # [[0.94497115]]
    # [[-0.01895357]]
    print(PAIRWISE_METRIC(starwars1_vec, starwars2_vec))
    print(PAIRWISE_METRIC(starwars1_vec, starwars3_vec))
    print(PAIRWISE_METRIC(starwars4_vec, starwars5_vec))
    print(PAIRWISE_METRIC(starwars4_vec, starwars6_vec))
    # [[0.5414141]]
    # [[-0.31255373]]
    # [[0.69569991]]
    # [[0.66382108]]
    '''

    ## Some simple testing of cosine similarity on user 22085
    '''
    user_vector = user_vectors_dict[22085]
    user_vector = user_vector.reshape(1,-1)  # sklearn cos_sim need 2D
    print(user_vector)

    user_movie_IDs = dict_groups_trg[22085]
    # user_movie_IDs = dict_groups_val[22085]
    print(user_movie_IDs)

    user_movie_vectors = movie_vectors_df.loc[user_movie_IDs].to_numpy()
    print(user_movie_vectors)

    cos_sim = PAIRWISE_METRIC(user_movie_vectors, user_vector)
    print(cos_sim)
    print(len(user_movie_IDs), len(user_movie_vectors), len(cos_sim))
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
        df_eval[[LIKED, 'scores']].to_csv('Model/word2vec_pyspark_0_{}_{}.csv'.format(
                                            EVAL_DATASET, PAIRWISE_METRIC.__name__))
        with open('Model/word2vec_pyspark_0_{}_{}.pickle'.format(
                    EVAL_DATASET, PAIRWISE_METRIC.__name__), 'wb') as f:
            pickle.dump(df_eval, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        df_eval[[LIKED, 'scores']].to_csv('Model/word2vec_pyspark_0_{}_dot.csv'.format(
                                            EVAL_DATASET))
        with open('Model/word2vec_pyspark_0_{}_dot.pickle'.format(
                    EVAL_DATASET), 'wb') as f:
            pickle.dump(df_eval, f, protocol=pickle.HIGHEST_PROTOCOL)

    mask_not_nan = np.logical_not(np.isnan(scores)) # ~ may also work
    truth = df_eval[LIKED].to_numpy()
    print(truth[mask_not_nan].min(), truth[mask_not_nan].max())
    print(scores[mask_not_nan].min(), scores[mask_not_nan].max())
    print(roc_auc_score(truth[mask_not_nan], scores[mask_not_nan]))
    ## --------------------------------------------------------------------



    '''
    ## some pyspark testing...

    nums_RDD0 = sc.parallelize([1,2,3,4])   # <class 'pyspark.rdd.RDD'>
    nums_RDD1 = nums_RDD0.map(lambda x: x*x)# <class 'pyspark.rdd.PipelinedRDD'>
    nums_list = nums_RDD1.collect()         # <class 'list'>
    print(type(nums_RDD0), type(nums_RDD1), type(nums_list))
    print(nums_RDD0.take(10))
    print(nums_RDD1)
    print(nums_list)


    row = Row("val") # Or some other column name
    nums_DF0 = nums_RDD0.map(row).toDF()     # <class 'pyspark.sql.dataframe.DataFrame'>
    nums_DF1 = nums_RDD1.map(row).toDF()     # <class 'pyspark.sql.dataframe.DataFrame'>
    print(type(nums_DF0), type(num_DF1))
    nums_DF0.show()
    nums_DF1.show()
    '''

    # sc.stop()


if __name__ == '__main__':
    start_time = datetime.now()

    run_validation_metrics()

    end_time = datetime.now()
    run_time = end_time - start_time
    print('Total Run Time: {}'.format(run_time))
