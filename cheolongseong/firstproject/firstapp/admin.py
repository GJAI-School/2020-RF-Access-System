from django.contrib import admin
from .models import Embedding, AccessHistory

# Register your models here.
admin.site.register(Embedding)
admin.site.register(AccessHistory)