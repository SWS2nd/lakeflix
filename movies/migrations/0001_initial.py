# Generated by Django 4.0.1 on 2022-01-27 12:59

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Movies',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title_kor', models.CharField(max_length=100)),
                ('director', models.CharField(max_length=200)),
                ('year', models.CharField(max_length=10)),
                ('genre', models.CharField(max_length=200)),
                ('play_time', models.CharField(max_length=20)),
                ('justwatch_rating', models.CharField(max_length=10)),
                ('imdb_rating', models.CharField(max_length=10)),
                ('synopsis', models.TextField(max_length=500)),
                ('poster', models.CharField(max_length=100)),
            ],
            options={
                'db_table': 'movie',
            },
        ),
    ]
