Team 5 - Movie Edge

===Description===
Movie Edge is a movie recommendation tool that uses neural network "word" embeddings to both render movies in "taste space" as well as provide predictions to users.

===DEMO Video of below Installation section===
https://youtu.be/-VUsqzLVf6k

===Installation===
0. Don't even bother installing (cumbersome) and simply go to: http://rockofmaine4989.pythonanywhere.com/movie_edge/ or http://rockofmaine4989.pythonanywhere.com/movie_edge/no_graphic/ . We like the pretty one with graphics (first one).

- Install Python 3.7 from python.org
- Navigate to the CODE folder in the same directory as this file.
- If you want to create a new environment for your python installation, execute $python3.7 -m venv ENV
- This will create a new environment called "ENV".
- Activate this environment via $source ENV/bin/activate
- You should probably do 
	- $pip install Cython
	- $pip install wheel
	- This is due to the issue noted below
- Navigate to CODE directory (if you aren't there already), and there should be files called "requirements.txt" and "requirements-slim.txt" there. The "slim" file is okay to use if you are doing this installation. The non-slim version will be needed if you want to do a full build as detailed in the following section.
- Execute $pip install -r requirements.txt or $pip install -r requirements-slim.txt
	- IF YOU RUN INTO Cython issues: then $pip install Cython and then try $pip install -r requirements.txt again. Cython simply needs to be installed first almost always. Command noted here for Cython: https://pypi.org/project/Cython/
	- IF YOU RUN INTO ERRORS WITH fitsne, remove the line from the requirements.txt file. It is only needed if you want to rebuild fully from scratch. To run the webapp, it is not needed. It requires manual compilation of FFTW and likely is more than you have any desire to do.
- Go to the "cse6242_team5" directory
- Execute $python manage.py runserver. If in CODE/cse6242_team5/cse6242_team5/settings.py, debug=False (~line 26) change it to True.
- Navigate to 127.0.0.1:8000/movie_edge/ or 127.0.0.1:8000/movie_edge/no_graphic/
- Django sometimes throws "ConnectionResetError: [Errno 54] Connection reset by peer". If the site doesn't load, try again. Make sure you're not going to 127.0.0.1:8000/ You need to go to the movie_edge site within at: 127.0.0.1:8000/movie_edge/ as stated above.
- Rate some movies!
- Things to understand: we've kindly packaged our database and neural network embedding with the project to avoid you needing to build it from scratch. If you want to, please read below.


==Installation / Full Build Detailed (you probably don't want to do this)==
- Ensure you've activated the Python3.7 environment built in the "===Installation===" section and that all the packages have been successfully installed into. fitsne is a pain.

=Build Database=
To fully rebuild our data from scratch, it is more involved...
- Go to the cse6242_team5 directory (this is a Django project). The following three commands will create a empty database
- Delete db.sqlite3 as it's the database we've kindly pre-populated for you.
- Run $python manage.py makemigrations movie_edge
- Run $python manage.py migrate movie_edge
- Now you have an empty database. We need to populate it. But first, we must generate a Word2Vec model...

=Acquire data=
- This was done piecemeal by our team on multiple machines, so bubble gum and duct tape may be needed
- Go to https://grouplens.org/datasets/movielens/ and download the link here: http://files.grouplens.org/datasets/movielens/ml-20m.zip. Extract it. It will just be a folder called ml-20m. This ml-20m folder is expected to be in the CODE directory.
- Now we need to get the metadata.
- Now you will need to go to http://www.omdbapi.com/ and become a patron. Get your API key and enter it in omdb_scrapper_all_files.py at line 18. You can use a free API key, but you will only be able to download 1000 movies a day. Modify the scraper file appropriately if you wish to do this. 
- Run: $python omdb_scrapper_all_files.py
	- This file is located in CODE.
- Run: $python metadata.py
	- Generates metadata.pkl. 
- Run: $python convert.py 
	- Binarize the data in ratings.csv
	- this will produce a file "binarized.hdf" 
- Run $python split.py
	- Generates the train (trg), validation (val) and test (tst) datasets. 

=Process data= 
- Go to the root CODE directory
- Run $jupyter notebook 
	- Must be in the environment you $pip install'd into earlier.
- Open gensim_word2vec.ipynb and "Cells -> Run All". This will generate the gensim Word2vec model, titled: w2v_vs_64_sg_1_hs_1_mc_1_it_4_wn_32_ng_2_all_data_trg_val_tst.gensim
- Move the model into the gensim_models2 directory (it will originally be written into the same directory as the Jupyter notebook, CODE). You will copy over (aka, delete) the existing model in the directory that we've packaged in the ZIP for you.
- Run $python makepayload.py
	- Running the script requires use of the FFT based t-SNE, so you will need to install Flt-SNE from here: https://github.com/KlugerLab/FIt-SNE which also requires downloading FFTW by following the directions on their site: http://fftw.org/. This will require you to keep the variable `fast` = True.
	- Installation will require a  C compiler on Linux/Mac.
	- On Windows, you will have to find the precompiled binaries here: https://github.com/KlugerLab/FIt-SNE/releases/download/v1.1.0/FItSNE-Windows-1.1.0.zip and follow the PATH setup instructions from https://github.com/KlugerLab/FIt-SNE
	- If you don't want to install anything this low level, set fast to False (this is moderately untested. We used the FFT method as detailed in the report.)
	- You may need to add the Flt-SNE repo to your system path using a line like so before the import in the code
		- sys.path.append('/Users/some/path/to/FIt-SNE')
- We recommend executing the cse6242_team5/db_NULL_fix.sql on the db.sqlite3 file to fix some nulls in the database.
	- Execute this via the SQLite shell or your preferred Sqlite DB console.
- You now have a populated db.sqlite3 file!
- Navigate to the cse6242_team5 directory
- From here, go ahead and execute $python manage.py runserver using the simple "===Installation===" directions above.


= Important =
- metadata.pkl, and the db.sqlite3 files have already been prepared for you. There is no need to perform the above steps. 


===Execution===

See step 5 and 6 in the "Installation" (not detailed) version. When you navigate to the pages, that's it! See the demo video if unclear... or just go to http://rockofmaine4989.pythonanywhere.com/movie_edge/.

