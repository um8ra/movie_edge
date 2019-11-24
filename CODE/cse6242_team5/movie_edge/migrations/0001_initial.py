# Generated by Django 2.2.6 on 2019-11-20 05:29

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='c0',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cluster_id', models.IntegerField(db_index=True)),
                ('cluster_size', models.IntegerField()),
                ('x', models.FloatField()),
                ('y', models.FloatField()),
                ('genres', models.CharField(max_length=512)),
                ('actors', models.CharField(max_length=512)),
                ('metascore', models.IntegerField(null=True)),
                ('imdb_rating', models.FloatField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='c1',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cluster_id', models.IntegerField(db_index=True)),
                ('cluster_size', models.IntegerField()),
                ('x', models.FloatField()),
                ('y', models.FloatField()),
                ('genres', models.CharField(max_length=512)),
                ('actors', models.CharField(max_length=512)),
                ('metascore', models.IntegerField(null=True)),
                ('imdb_rating', models.FloatField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='c2',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cluster_id', models.IntegerField(db_index=True)),
                ('cluster_size', models.IntegerField()),
                ('x', models.FloatField()),
                ('y', models.FloatField()),
                ('genres', models.CharField(max_length=512)),
                ('actors', models.CharField(max_length=512)),
                ('metascore', models.IntegerField(null=True)),
                ('imdb_rating', models.FloatField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='c3',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cluster_id', models.IntegerField(db_index=True)),
                ('cluster_size', models.IntegerField()),
                ('x', models.FloatField()),
                ('y', models.FloatField()),
                ('genres', models.CharField(max_length=512)),
                ('actors', models.CharField(max_length=512)),
                ('metascore', models.IntegerField(null=True)),
                ('imdb_rating', models.FloatField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='c4',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cluster_id', models.IntegerField(db_index=True)),
                ('cluster_size', models.IntegerField()),
                ('x', models.FloatField()),
                ('y', models.FloatField()),
                ('genres', models.CharField(max_length=512)),
                ('actors', models.CharField(max_length=512)),
                ('metascore', models.IntegerField(null=True)),
                ('imdb_rating', models.FloatField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Movie',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('embedder', models.CharField(max_length=128)),
                ('movie_id', models.IntegerField(db_index=True)),
                ('movie_title', models.CharField(max_length=256)),
                ('genres', models.CharField(max_length=128)),
                ('x', models.FloatField()),
                ('y', models.FloatField()),
                ('L0', models.IntegerField()),
                ('L1', models.IntegerField()),
                ('L3', models.IntegerField()),
                ('L2', models.IntegerField()),
                ('L4', models.IntegerField()),
                ('L5', models.IntegerField()),
                ('L0x', models.FloatField()),
                ('L0y', models.FloatField()),
                ('L1x', models.FloatField()),
                ('L1y', models.FloatField()),
                ('L2x', models.FloatField()),
                ('L2y', models.FloatField()),
                ('L3x', models.FloatField()),
                ('L3y', models.FloatField()),
                ('L4x', models.FloatField()),
                ('L4y', models.FloatField()),
                ('L5x', models.FloatField()),
                ('L5y', models.FloatField()),
                ('poster_url', models.CharField(max_length=256, null=True)),
                ('runtime', models.IntegerField(null=True)),
                ('director', models.CharField(max_length=256)),
                ('actors', models.CharField(max_length=512)),
                ('metascore', models.IntegerField(null=True)),
                ('imdb_rating', models.FloatField(null=True)),
                ('imdb_votes', models.IntegerField()),
            ],
        ),
        migrations.AddConstraint(
            model_name='movie',
            constraint=models.UniqueConstraint(fields=('embedder', 'movie_id'), name='unique_movie_per_embedding'),
        ),
    ]