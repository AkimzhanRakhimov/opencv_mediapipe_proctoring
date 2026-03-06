from django.db import models

# Create your models here.
class Userdata(models.Model):
    id=models.IntegerField(primary_key=True)
    sign=models.CharField(max_length=50)
    duration=models.IntegerField()
    created_at=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.id
