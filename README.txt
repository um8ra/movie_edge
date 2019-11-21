Team 5 - Movie Edge

===Description===
Movie Edge is a movie recommendation tool that uses neural network "word" embeddings to both render movies in "taste space" as well as provide predictions to users.

===Installation===
0. Don't even bother installing (cumbersome) and simply go to: http://rockofmaine4989.pythonanywhere.com/movie_edge/ or http://rockofmaine4989.pythonanywhere.com/movie_edge/no_graphic/ . We like the pretty one with graphics (first one).

1. Install Python 3.7 or higher
2. Navigate to the folder you find this file in
2. $pip install -r requirements.txt
	- IF YOU RUN INTO ERRORS WITH fitsne, remove the line from the requirements.txt file. It is only needed if you want to rebuild fully from scratch. To run the webapp, it is not needed. It requires manual compilation of FFTW and likely is more than you have any desire to do.
3. Go to the "cse6242_team5" directory
4. Execute $python manage.py runserver. If debug=False in settings, change it to True.
5. Navigate to 127.0.0.1:8000/movie_edge/ or 127.0.0.1:8000/movie_edge/no_graphic/
6. Rate some movies!
7. Things to understand: we've kindly packaged our database with the project to avoid you all needing to build it from scratch. If you want to, please read below.


==Installation Detailed==

=Build Database=
To fully rebuild our data from scratch, it is more involved...
- Go to the cse6242_team5 directory (this is a Django project). The following three commands will create a empty database
- Delete db.sqlite3 as it's the database we've kindly pre-populated for you.
- Run $python manage.py makemigrations movie_edge
- Run $python manage.py migrate movie_edge
- Now you have an empty database. We need to populate it. But first, we must generate a Word2Vec model...

=Acquire data=
- We've already put the data into the ZIP for you, and our group has pickled and HDF'd lots of the data but if you're adventerous, you can download the data and make minimal tweaks to the files.
- Go to https://grouplens.org/datasets/movielens/ and download the link here: http://files.grouplens.org/datasets/movielens/ml-20m.zip. This ml-20m folder is expected to be in the root directory of the project.
- One of our groupmembers divided this into training/test/validation, so scripts may need to be tweaked in order to read from CSV rather than binary blobs. Anywhere in the code you see 'binarized.hdf' this is a reference to OMDB ml-20m data.
- Metadata for the movies was also downloaded via API from here: https://www.omdbapi.com/. Anywhere you see 'metadata.pkl' referenced in the code, this is OMDB API data.

=Process data= 
- Go up one directory to "cse6242_project" 
- Run $jupyter notebook in the environment you $pip install'd into earlier.
- Open gensim_word2vec.ipynb and "run all cells". This will generate the gensim Word2vec model.
- Move the model into the gensim_models2 directory (it will be written into the same directory as the Jupyter notebook)
- Then you should open the makepayload.ipynb and "run all cells" but first:
	- If you want to run the FFT based t-SNE, you will need to download Flt-SNE from here: https://github.com/KlugerLab/FIt-SNE which also requires downloading FFTW by following the directions on their site: http://fftw.org/. This will require you to keep the variable `fast` = True.
	- If you don't want to install anything this low level, set fast to false (this is moderately untested. We used the FFT method as detailed in the report.)
	- Now run all cells :)
- You now have a populated db.sqlite3 file!
- From here, go ahead and execute $python manage.py runserver using the simple "===Installation===" directions above.


===Execution===

See step 5 and 6 in the "Installation" (not detailed) version. When you navigate to the pages, that's it!

