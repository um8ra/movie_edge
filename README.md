# cse6242_project
CSE 6242 Team Project

# Workflow

Caveat: Some of the paths use hdf files, which were made by Jon Tay. These references in the code can be easily changed to the movielens CSVs. The hdf files we are using are split into train/test/validation sets, so results may vary based on the percentage splits used. The `df_trg` and other files referencing `binarized.hdf` are Multi-Indexed on `userId` and `movieId`. 

1. Download and extract the ml-20m.zip from https://grouplens.org/datasets/movielens/ into this Git repo
    * This is your source data for everything following
    * It should be in a folder called `ml-20m` that is inside this Git repo
2. You should create and install a Python 3 environment via PIP or Conda
    * Requirements file is requirements.txt
    * Can be installed via `$pip install -r requirements.txt`
3. Execute the `gensim_word2vec.ipynb` notebook
    * Change the gridsearch parameters as you see fit
    * We've found `size` < 32 to be good and t-SNE-able
    * Windowing helps speed up training
    * Skip-gram seems to have a *huge* impact on accuracy (for the better)
    * Run all cells!
4. The above direction writes out a bunch of `.gensim` files. These are the saved models for the next step
5. Use the `model_tester.ipynb` file to evaluate models.
    * We've found higher average similarity scores to yield better results.
    * Pick a model (filename) to use for the next step
6. Use the `tsne_gensim.ipynb` file to run t-SNE on the word vectors contained in your model of choice
    * The output will also be plot using bokeh
    * Enjoy!