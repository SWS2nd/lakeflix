# Generated by Django 4.0.2 on 2022-02-03 07:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('movies', '0002_alter_movies_genre_list'),
    ]

    operations = [
        migrations.AlterField(
            model_name='movies',
            name='genre_list',
            field=models.ManyToManyField(to='movies.MovieType'),
        ),
    ]
