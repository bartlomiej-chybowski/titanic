from django.shortcuts import render


def home(request):
    return render(request, 'index.html')


def get_predictions(pclass, sex, age, sibsp, parch, fare, C, Q, S):
    import pickle
    model = pickle.load(open("titanic_survival_ml_model.sav", "rb"))
    scaled = pickle.load(open("scaler.sav", "rb"))
    prediction = model.predict(sc.transform([[pclass, sex, age, sibsp, parch, fare, C, Q, S]]))

    if prediction == 0:
        return "not survived"
    elif prediction == 1:
        return "survived"
    else:
        return "error"


def result(request):
    pclass = int(request.GET['pclass'])
    sex = int(request.GET['sex'])
    age = int(request.GET['age'])
    sibsp = int(request.GET['sibsp'])
    parch = int(request.GET['parch'])
    fare = int(request.GET['fare'])
    embC = int(request.GET['embC'])
    embQ = int(request.GET['embQ'])
    embS = int(request.GET['embS'])

    result = get_predictions(p, sex, age, sibsp, parch, fare, embC, embQ, embS)

    return render(request, 'result.html', {'result':result})