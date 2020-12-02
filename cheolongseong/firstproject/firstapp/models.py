from django.db import models

# Create your models here.
class Embedding(models.Model):
    name = models.CharField(max_length=100)
    rfid = models.CharField(max_length=200)
    face_embedding = models.CharField(max_length=200)
    vein_embedding = models.CharField(max_length=200)

class AccessHistory(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    rfid = models.ForeignKey(Embedding, on_delete=models.DO_NOTHING)
    face_check = models.CharField(max_length=100)
    temp = models.FloatField()
    vein_check = models.CharField(max_length=100)