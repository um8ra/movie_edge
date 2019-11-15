const padding = 25;
const MOVIE_ID = 'movie_id';
const MOVIE_TITLE = 'movie_title';
const DIRECTOR = 'director';
const ACTORS = 'actors';
const TITLE = 'title';
const GENRES = 'genres';
const CLUSTER = 'cluster';
const COLOR = 'color';
const POSTER_URL = 'poster_url';
const MOVIE_CHOICES = 'movie_choices';
const LIKE = 'movies_liked';
const DISLIKE = 'movies_disliked';
const MOVIES_SHOWN = 'movies_shown';
const X = 'x';
const Y = 'y';
const moviesLiked = new Set(); // using set since it will do deduplication
const moviesDisliked = new Set(); // using set since it will do deduplication
const zoomParams = {
    0: {r: 20, w: 2},
    1: {r: 10, w: 1},
    2: {r: 5, w: 0.5},
    3: {r: 2.5, w: 0.25},
    4: {r: 1.25, w: 0.125},
    5: {r: 0.5, w: 0.05},
    minZoom: 1,
    maxZoom: 64,
};
const gridID = 'grid';
const bbox_pad = 0;
const IMDB_RATING = 'imdb_rating';
const QUEUE_MAX_LENGTH = 10;
let currentGrid = Array();
const gridHistorySet = new Set();
const moviesLikedQueue = Array();
const moviesDislikedQueue = Array();