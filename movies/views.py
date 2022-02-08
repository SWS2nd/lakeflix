import csv
import json

from . import models
from django.shortcuts import redirect, render
from django.urls import reverse
from django.http import Http404
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required

import random

# 추천 알고리즘 구현에 필요한 모듈들
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.lib.function_base import percentile
# python ast 내장 모듈의 literal_eval() 메소드를 이용하기 위함
import ast


@login_required(login_url="/users/login/")
def home_page(request):
    all_movies = models.Movies.objects.all()
    random_movie = random.choice(all_movies)
    action_movies = models.Movies.objects.filter(genre_list__name="액션")[:12]
    user = request.user.is_authenticated
    if user:
        return render(request, "movies/all_movies.html", context={"movies": all_movies, 'random_movie': random_movie, 'action_movies': action_movies})
    else:
        return redirect(reverse("users:login"))

# @login_required(login_url="/users/login/") =  로그인 안했으면 로그인페이지로 돌리기


@login_required(login_url="/users/login/")
def test(request):
    user = request.user.is_authenticated
    if user:
        return render(request, "movies/subpagetest.html")
    else:
        return redirect(reverse("users:login"))


@login_required(login_url="/users/login/")
def movie_detail(request, pk):
    try:
        movie = models.Movies.objects.get(pk=pk)
        # movie QuerySet을 content_based_recommendation 메소드의 인자로 넣어,
        # QuerySet 혹은 Json 형식을 return 받아 recommend_movies 변수에 저장
        recommend_movies = content_based_recommendation(movie)
        return render(request, "movies/detail.html", context={"movie": movie, "recommend_movies": recommend_movies})

    except models.Movies.DoesNotExist:
        raise Http404()


def content_based_recommendation(movie):
    movies_df = pd.read_csv('static/JustWatch_dataset(imdb_change).csv', encoding='cp949')
    # print(movies_df)

    # 필요한 컬럼만 추출해서 새로운 데이터 프레임 생성
    movies_df = movies_df[
        ['title_kor', 'year', 'genre_list', 'play_time', 'director', 'justwatch_rating', 'imdb_rating',
         'imdb_rating_average', 'imdb_vote_count', 'synopsis']]

    # 한 행만 바꾸는게 아닌 데이터 프레임의 한 열을 바꾸어야 하기에 pandas의 apply 메소드를 활용하여 해당 열의 모든 값을 변경해 준다.
    movies_df['genre_list'] = movies_df['genre_list'].apply(ast.literal_eval)
    # print(movies_df['genre_list'][5], type(movies_df['genre_list'][5]))

    # 해당 리스트를 countvectorizer 적용을 위해 공백문자로 word 단위로 구분되는 문자열로 변환
    movies_df['genre_literal'] = movies_df['genre_list'].apply(lambda x: (' ').join(x))
    # print(movies_df['genre_literal'])
    count_vect = CountVectorizer(min_df=0, ngram_range=(1, 2))
    genre_mat = count_vect.fit_transform(movies_df['genre_literal'])
    # print(genre_mat.shape)

    # 코사인 유사도로 장르간 유사도 측정
    genre_sim = cosine_similarity(genre_mat, genre_mat)
    # print(genre_sim.shape)
    # print(genre_sim)

    # argsort( )[ :, ::-1]을 이용하면 유사도가 높은 순으로 정리된 genre_sim 객체의 비교 행 위치 인덱스 값을 얻을 수 있음
    genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]
    # print(genre_sim_sorted_ind[:1])

    # --- 가중 평점 계산 ---

    # imdb_rating 컬럼의 값들이 None, 6.5, 6.5(1k), 6.5(1m) 과 같은 식으로 4가지 타입이 존재하며,
    # None : 평가 없음 => 1.0, 1표 투표로 대체하겠음 ==> 1.0 (1)
    # 6.5 : vote_count 없음 => 1표 투표로 대체하겠음 ==> 6.5 (1)
    # 6.5(1k) : k의 경우 원래는 x1000 이나, => x10으로 대체하겠음 ==> 6.5 (10)
    # 6.5(1m) : m의 경우 원래는 x백만 이나, => x10000으로 대체하겠음 ==> 6.5 (10000)

    # 전체 투표 횟수에서 상위 60%에 해당하는 횟수를 기준으로 정한 상태에서 movies_df의 apply() 함수의 인자로 입력해 가중 평점 계산
    percentile_factor = 0.6
    m = movies_df['imdb_vote_count'].quantile(percentile_factor)
    C = movies_df['imdb_rating_average'].mean()

    # 가중 평점( Weighted Rating ) = (v/(v+m) * R + (m/(v+m)) * C
    # v: 개별 영화에 평점을 투표한 횟수, m : 평점을 부여하기 위한 최소 투표 횟수,
    # R : 개별 영화에 대한 평균 평점, C : 전체 영화에 대한 평균 평점.
    # imdb_vote_average를 imdb_vote_count를 반영한 가중 평점으로 바꾸는 함수 생성
    def weighted_vote_average(record):
        v = record['imdb_vote_count']
        R = record['imdb_rating_average']

        return ((v / (v + m)) * R) + ((m / (m + v)) * C)

    # imdb_weighted_vote 컬럼 추가
    movies_df['imdb_weighted_vote'] = movies_df.apply(weighted_vote_average, axis=1)

    # 장르 유사성이 높은 영화를 top_n의 2배수만큼 후보군으로 선정한 뒤에 weighted_vote 칼럼 값이 높은 순으로 top_n만큼 추출하는 방식
    def find_sim_movie_weighted_vote(df, sorted_ind, title_name, top_n=10):
        title_movie = df[df['title_kor'] == title_name]
        title_index = title_movie.index.values

        similar_indexes = sorted_ind[title_index, :(top_n * 2)]
        similar_indexes = similar_indexes.reshape(-1)
        similar_indexes = similar_indexes[similar_indexes != title_index]

        return df.iloc[similar_indexes].sort_values('imdb_weighted_vote', ascending=False)[:top_n]

    # 영화 제목이 들어왔을 때 해당 영화와 평점 가중치로 장르별 유사도를 측정하여 상위 12개의 영화를 추출
    similar_movies_weighted_vote = find_sim_movie_weighted_vote(movies_df, genre_sim_sorted_ind, movie.title_kor, 12)
    # print(type(similar_movies_weighted_vote), similar_movies_weighted_vote)
    # print(similar_movies_weighted_vote['title_kor'])

    # 인덱스 값에 +1 해야 해당 영화의 pk가 됨.(인덱스는 0부터 시작!)
    similar_movies_weighted_vote.index.name = "index"
    # 해당 클래스는 DataFrame이며, Int64Index 클래스임
    # print(type(similar_movies_weighted_vote), similar_movies_weighted_vote.index)

    # 따라서, 리스트로 변환 후 각 요소에 1씩 더한 값을 movie_pk_list 변수에 저장
    index_list = list(similar_movies_weighted_vote.index)
    # print(type(index_list), index_list)
    movie_pk_list = list(map(lambda a: a + 1, index_list))
    print(type(movie_pk_list), "movie_pk_list : ", movie_pk_list)

    recommend_movies = models.Movies.objects.filter(pk__in=movie_pk_list)  ## __in 으로 리스트 안의 요소들로 필터링 가능
    print(type(recommend_movies),"recommend_movies : ", recommend_movies) # recommend_movies == QuerySet 타입

    # QuerySet 타입을 json으로 넘길 수 있도록 안의 값을 빼서 리스트화.
    # context = {'recommend_movies': list(recommend_movies.values())}
    # print(type(context), context)

    return recommend_movies


def tag_search(request):
    all_movies = models.Movies.objects.all()
    all_genres = models.MovieType.objects.all()
    mov_cnt = len(all_movies)
    if request.method == 'GET':
        return render(request, 'movies/tag_search.html', context={'movies': all_movies, "genres": all_genres, "mov_cnt": mov_cnt})
    elif request.method == 'POST':
        checked_movie_type = request.POST.get('movie_type', '') # 체크한 영상의 타입
        filter_attr = request.POST.get('filter_attr', '') # 체크한 영상을 필터링 or 제외할 것인지 값
        print('checked_movie_type =',checked_movie_type, '/ filter_attr =',filter_attr)
        if filter_attr == 'filter':
            # 체크된 태그들만 보여줘. = objects.filter() -> 괄호 안의 값으로 필터링 해 불러옴
            filtered_movies = models.Movies.objects.filter(genre_list__name=checked_movie_type)
        elif filter_attr == 'exclude':
            # 체크된 태그들만 빼고 보여줘 = objects.exclude() 사용 -> 괄호 안의 값을 제외하고 불러옴
            filtered_movies = models.Movies.objects.exclude(genre_list__name=checked_movie_type)
        else: # 위 2개의 필터 속성을 제외한 값이 들어올 경우
            # 전부 다 보여줘 = all_movies
            filtered_movies = all_movies

        # QuerySet 타입을 json으로 넘길 수 있도록 안의 값을 빼서 리스트화.
        context = {'movies': list(filtered_movies.values())}
        # print(type(context), context)

        return HttpResponse(json.dumps(context), content_type='application/json')
        # return render(request, 'movies/tag_search.html', context={'movies': filtered_movies, "genres": all_genres})


def csv_test(request):
    CSV_PATH = "static/JustWatch_dataset(imdb_change).csv"
    with open(CSV_PATH, newline='') as csvfile:
        data_reader = csv.DictReader(csvfile)

        for row in data_reader:

            genre_data = row['genre_list']
            # genre_data_list 수정 필요 너무 replace 불필요하게 많이 사용함
            genre_data_list = genre_data.replace("[", "").replace(
                "]", "").replace(" ", "").replace("'", "").split(",")

            movies, _ = models.Movies.objects.get_or_create(
                title_kor=row['title_kor'],
                year=row['year'],
                play_time=row['play_time'],
                director=row['director'],
                justwatch_rating=row['justwatch_rating'],
                imdb_rating=row['imdb_rating'],
                imdb_rating_average=row['imdb_rating_average'],
                imdb_vote_count=row['imdb_vote_count'],
                synopsis=row['synopsis'],
                poster=row['poster_src'],
            )

            # 데이터 셋에 장르를 걸러서 장르 타입에 올린다
            for genre in genre_data_list:
                MovieGenre, _ = models.MovieType.objects.get_or_create(
                    name=genre
                )
                movies.genre_list.add(MovieGenre)

    return HttpResponse("여기는 데이터 셋을 DB에 업로드 하는 곳입니다! 업로드 할때를 제외하고는 와서는 안되는 페이지 입니다!! 여기로 올시 경민님께 말씀해주세요!")
