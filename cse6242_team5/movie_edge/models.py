from django.db import models


class Movie(models.Model):
    embedder = models.CharField(max_length=128, db_index=True, null=False)  # The model that ran this data
    movie_id = models.IntegerField(null=False)  # Movie ID based on movielens data
    movie_title = models.CharField(max_length=256, null=False)
    genres = models.CharField(max_length=128, null=False)
    x = models.FloatField(null=False)  # X value post t-SNE
    y = models.FloatField(null=False)  # Y value post t-SNE
    cluster = models.IntegerField(null=False)  # Agglomerative clustering pre-t-SNE
    mean = models.FloatField(null=False)
    std = models.FloatField(null=True)  # nullable in case only one rating
    count = models.IntegerField(null=False)
    poster_url = models.CharField(max_length=256, null=False)
    runtime = models.IntegerField(null=True)
    director = models.CharField(max_length=256, null=False)
    actors = models.CharField(max_length=512, null=False)
    metascore = models.IntegerField(null=True)
    imdb_rating = models.FloatField(null=True)
    imdb_votes = models.IntegerField(null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['embedder', 'movie_id'], name='unique_movie_per_embedding'),
        ]
