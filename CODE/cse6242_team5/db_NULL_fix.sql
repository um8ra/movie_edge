UPDATE movie_edge_movie
SET imdb_votes = -1
WHERE imdb_votes = 'null';

UPDATE movie_edge_movie
SET metascore = NULL
WHERE metascore = 'null';

UPDATE movie_edge_movie
SET imdb_rating = NULL
WHERE imdb_rating = 'null';
