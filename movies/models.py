from core import models as core_models
from django.db import models
from django.urls import reverse


class AbstractItem(core_models.TimeStampedModel):
    """ Abstract Item """
    name = models.CharField(max_length=80, null=True, blank=True)

    class Meta:
        abstract = True
        # 이 클레스는 데이터 베이스에 올라가지 않게된다 movie모델에서 many to many에서 부가적으로만 사용되는 용도이기 때문에 설정해주는것

    def __str__(self):
        return self.name  # admin 페이지에서 각각 이름 주기 (필수 아님)


class MovieType(AbstractItem):
    pass


class Movies(models.Model):
    """ Movies Model """
    class Meta:
        db_table = "movie"  # DB에 들어갈 이름 설정 (필수 아님)
        verbose_name = "movie"  # admin 페이지에서 앱이름 설정 (필수 아님)

    title_kor = models.CharField(max_length=100)
    year = models.CharField(max_length=10)
    play_time = models.CharField(max_length=20)
    director = models.CharField(max_length=200)
    justwatch_rating = models.CharField(max_length=10)
    imdb_rating = models.CharField(max_length=10)
    imdb_rating_average = models.CharField(max_length=10)
    imdb_vote_count = models.CharField(max_length=10)
    synopsis = models.TextField(max_length=500)
    poster = models.CharField(max_length=100)
    genre_list = models.ManyToManyField(
        MovieType)
    # blank=True 는 빈 채로 저장되는 것을 허용합니다. ㅡ

    def __str__(self):
        return self.title_kor  # admin 페이지에서 각 무비들 이름 주기 (필수 아님)

    def get_absolute_url(self):
        return reverse("movies:detail", kwargs={'pk': self.pk})
