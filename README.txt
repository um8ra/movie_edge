Team 5 - Movie Edge

===Description===
Movie Edge is a movie recommendation tool that uses neural network "word" embeddings to both render movies in "taste space" as well as provide predictions to users.

===Installation===
0. Don't even bother installing (cumbersome) and simply go to: http://rockofmaine4989.pythonanywhere.com/movie_edge/ or http://rockofmaine4989.pythonanywhere.com/movie_edge/no_graphic/ . We like the pretty one with graphics (first one).

1. Install Python 3.7 or higher
2. Navigate to the folder you find this file in
2. $pip install -r requirements.txt
3. Go to the "cse6242_team5" directory
4. Execute $python manage.py runserver. If debug=False in settings, change it to True.
5. Navigate to 127.0.0.1:8000/movie_edge/ or 127.0.0.1:8000/movie_edge/no_graphic/
6. Rate some movies!


==Installation Detailed==
To fully rebuild our data from scratch, it is more involved...
- Go to the cse6242_team5 directory (this is a Django project). The following three commands will create a empty database
- Delete db.sqlite3 as it's the database we've kindly pre-populated for you.
- Run $python manage.py makemigrations movie_edge
- Run $python manage.py migrate movie_edge
- Now you have an empty database. We need to populate it. But first, we must generate a Word2Vec model...
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

See step 5 and 6 above: when you navigate to the pages, that's it!

