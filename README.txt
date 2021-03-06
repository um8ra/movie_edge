Team 5 - Movie Edge

===Description===
Movie Edge is a movie recommendation tool that uses neural network "word" embeddings to both render movies in "taste space" as well as provide predictions to users.

===DEMO Video of below Installation section===
https://youtu.be/-XGvodphhEk

===Installation===
0. Don't even bother installing (cumbersome) and simply go to: http://rockofmaine4989.pythonanywhere.com/movie_edge/ or http://rockofmaine4989.pythonanywhere.com/movie_edge/no_graphic/ . We like the pretty one with graphics (first one).

- Install Python 3.7 or higher
- Navigate to the folder you find this file in
- If you want to create a new environment for your python installation, execute $python3.7 -m venv ENV
- This will create a new environment called "ENV".
- Activate this environment via $source ENV/bin/activate
- You should probably do $pip install Cython
	- This is due to the issue noted below
- Navigate to project root directory, there should be a file called "requirements.txt" in the root.
- Execute $pip install -r requirements.txt
	- IF YOU RUN INTO Cython issues: then $pip install Cython and then try $pip install -r requirements.txt again. Cython simply needs to be installed first almost always. Command noted here for Cython: https://pypi.org/project/Cython/
	- IF YOU RUN INTO ERRORS WITH fitsne, remove the line from the requirements.txt file. It is only needed if you want to rebuild fully from scratch. To run the webapp, it is not needed. It requires manual compilation of FFTW and likely is more than you have any desire to do.
- Go to the "cse6242_team5" directory
- Execute $python manage.py runserver. If debug=False in settings, change it to True.
- Navigate to 127.0.0.1:8000/movie_edge/ or 127.0.0.1:8000/movie_edge/no_graphic/
- Rate some movies!
- Things to understand: we've kindly packaged our database with the project to avoid you all needing to build it from scratch. If you want to, please read below.


==Installation / Full Build Detailed (you probably don't want to do this)==

=Build Database=
To fully rebuild our data from scratch, it is more involved...
- Go to the cse6242_team5 directory (this is a Django project). The following three commands will create a empty database
- Delete db.sqlite3 as it's the database we've kindly pre-populated for you.
- Run $python manage.py makemigrations movie_edge
- Run $python manage.py migrate movie_edge
- Now you have an empty database. We need to populate it. But first, we must generate a Word2Vec model...

=Acquire data=
- Go to https://grouplens.org/datasets/movielens/ and download the link here: http://files.grouplens.org/datasets/movielens/ml-20m.zip. This ml-20m folder is expected to be in the root directory of the project for most operations.
- Now we need to get the metadata. Copy the following files into the ml-20m folder: 
	- code: links2dl.py, omdb_scraper_all_files.py, metadata.py. convert.py, split.py
- Run links2dl.py. It will generate files f1.csv - f28.csv. 
- Now you will need to go to http://www.omdbapi.com/ and become a patron. Get your API key and enter it in omdb_scraper_all_files.py at line 18. You can use a free API key, but you will only be able to download 1000 movies a day. Modify the scraper file appropriately if you wish to do this. 
- Run omdb_scraper_all_files.py. It will generate f1.pkl - f28.pkl
- Now run metadata.py to generate metadata.pkl. 
- Run covert.py to binarize the data in ratings.csv
	- this will produce a file "binarized.hdf" 
- Use split.py to generate the train (trg), validation (val) and test (tst) datasets. 

=Process data= 
- Go to the root project directory: "cse6242_project" 
- Run $jupyter notebook in the environment you $pip install'd into earlier.
- Open gensim_word2vec.ipynb and "run all cells". This will generate the gensim Word2vec model.
- Move the model into the gensim_models2 directory (it will be written into the same directory as the Jupyter notebook)
- Then you should open the makepayload.ipynb and "run all cells" but first:
	- If you want to run the FFT based t-SNE, you will need to download Flt-SNE from here: https://github.com/KlugerLab/FIt-SNE which also requires downloading FFTW by following the directions on their site: http://fftw.org/. This will require you to keep the variable `fast` = True.
	- If you don't want to install anything this low level, set fast to false (this is moderately untested. We used the FFT method as detailed in the report.)
	- Now run all cells 
- You now have a populated db.sqlite3 file!
- From here, go ahead and execute $python manage.py runserver using the simple "===Installation===" directions above.


= Important =
- metadata.pkl, and the db.sqlite3 files have already been prepared for you. There is no need to perform the above steps. 


===Execution===

See step 5 and 6 in the "Installation" (not detailed) version. When you navigate to the pages, that's it! See the demo video if unclear... or just go to http://rockofmaine4989.pythonanywhere.com/movie_edge/.

