from django.contrib import admin
from . import models


@admin.register(models.Movies)
class moviesAdmin(admin.ModelAdmin):
    """ Custom User Admin """
    list_display = ("__str__", "title_kor")
