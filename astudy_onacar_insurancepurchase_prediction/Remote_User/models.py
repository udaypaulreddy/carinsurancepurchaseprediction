from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

class Car_Insurance(models.Model):

    idno= models.CharField(max_length=300)
    Gender= models.CharField(max_length=300)
    Age= models.CharField(max_length=300)
    Driving_License= models.CharField(max_length=300)
    Region_Code= models.CharField(max_length=300)
    Previously_Insured= models.CharField(max_length=300)
    Vehicle_Age= models.CharField(max_length=300)
    Vehicle_Damage= models.CharField(max_length=300)
    Annual_Premium= models.CharField(max_length=300)
    Policy_Sales_Channel= models.CharField(max_length=300)
    Vintage= models.CharField(max_length=300)
    IResponse= models.CharField(max_length=300)


class Car_Insurance_Prediction(models.Model):


    Gender= models.CharField(max_length=300)
    Age= models.CharField(max_length=300)
    Driving_License= models.CharField(max_length=300)
    Region_Code= models.CharField(max_length=300)
    Previously_Insured= models.CharField(max_length=300)
    Vehicle_Age= models.CharField(max_length=300)
    Vehicle_Damage= models.CharField(max_length=300)
    Annual_Premium= models.CharField(max_length=300)
    Policy_Sales_Channel= models.CharField(max_length=300)
    Vintage= models.CharField(max_length=300)
    IPrediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



