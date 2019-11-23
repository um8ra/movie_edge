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
- This was done piecemeal by our team on multiple machines, so bubblegum and duct tape may be needed
- We've already put the data into the db.sqlite3 for you, but if you're adventerous, you can download the data and make minimal tweaks to the files.
- Go to https://grouplens.org/datasets/movielens/ and download the link here: http://files.grouplens.org/datasets/movielens/ml-20m.zip. Unzip it. This ml-20m folder is expected to be in the root CODE directory of the project for most operations.
- Metadata for the movies was also downloaded via API from here: https://www.omdbapi.com/. Anywhere you see 'metadata.pkl' referenced in the code, this is OMDB API data.
- To download metadata:
	Links Data File Structure (links.csv in the ml-20m)
	Identifiers that can be used to link to other sources of movie data are contained in the file links.csv. Each line of this file after the header row represents one movie, and has the following format:
	movieId,imdbId,tmdbId
	movieId is an identifier for movies used by https://movielens.org. E.g., the movie Toy Story has the link https://movielens.org/movies/1.
	imdbId is an identifier for movies used by http://www.imdb.com. E.g., the movie Toy Story has the link http://www.imdb.com/title/tt0114709/.
	tmdbId is an identifier for movies used by https://www.themoviedb.org. E.g., the movie Toy Story has the link https://www.themoviedb.org/movie/862.
	See: http://files.grouplens.org/datasets/movielens/ml-20m-README.html
- With that in mind, the metadata.py script can be tweaked to use the links.csv. A teammate divvied up the file to work with this Python script, so the links.csv file won't work exactly.
- One of our groupmembers divided the ml-20m/movies.csv into training/test/validation, so scripts may need to be tweaked in order to read from CSV rather than binary blobs. Anywhere in the code you see 'binarized.hdf' this is a reference to OMDB ml-20m data, albeit chunks of it split at random (70/15/15).
- You should now have both core data (ml-20m/<many_files>), metadata.pkl. It's time to process some data!

=Process data= 
- Go to the root project directory: "cse6242_project" 
- Run $jupyter notebook in the environment you $pip install'd into earlier.
- Open gensim_word2vec.ipynb and "run all cells". This will generate the gensim Word2vec model.
- Move the model into the gensim_models2 directory (it will originally be written into the same directory as the Jupyter notebook)
- Then you should open the makepayload.ipynb and "run all cells" but first:
	- If you want to run the FFT based t-SNE, you will need to download Flt-SNE from here: https://github.com/KlugerLab/FIt-SNE which also requires downloading FFTW by following the directions on their site: http://fftw.org/. This will require you to keep the variable `fast` = True.
	- You may also need to download a C compiler to get this to work
	- If you don't want to install anything this low level, set fast to false (this is moderately untested. We used the FFT method as detailed in the report.)
	- Now run all cells :)
- We recommend executing the cse6242_team5/db_NULL_fix.sql file to fix some nulls in the database.
- You now have a populated db.sqlite3 file!
- From here, go ahead and execute $python manage.py runserver using the simple "===Installation===" directions above.


===Execution===

See step 5 and 6 in the "Installation" (not detailed) version. When you navigate to the pages, that's it! See the demo video if unclear... or just go to the rockofmaine site.

