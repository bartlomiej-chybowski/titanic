from django.shortcuts import render
import numpy as np
from Prediction_Web.Titanic.views.predictor import Predictor


def home(request):
    return render(request, 'index.html')


def get_predictions(pclass, sex, age, sibsp, parch, fare, embarked):
    predictor = Predictor('Titanic/titanic.csv')
    return predictor.predict(np.array([pclass, sex, age, sibsp, parch, fare, embarked]))


def result(request):
    pclass = int(request.GET['pclass'])
    sex = str(request.GET['sex'])
    age = float(request.GET['age'])
    sibsp = float(request.GET['sibsp'])
    parch = float(request.GET['parch'])
    fare = float(request.GET['fare'])
    embarked = str(request.GET['embarked'])

    return render(request, 'result.html',
                  {'result': get_predictions(pclass, sex, age, sibsp, parch, fare, embarked)})


def train(request):
    predictor = Predictor('Titanic/titanic.csv')
    predictor.train()

    return render(request, 'train.html')
