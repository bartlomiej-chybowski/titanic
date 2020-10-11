# Titanic
Simple classification application in django. 

Based on https://towardsdatascience.com/creating-a-machine-learning-based-web-application-using-django-5444e0053a09

## Endpoints
| URL | Result |
| --- | ------ |
| / | Main page with form |
| /train | Train model and save in `joblib` object together with scaler, and column transformer |
| /result | Result of prediction |


## Machine Learning training steps
1. Preprocessing
    1. Take care of missing numerical data
    2. Take care of missing categorical data
    3. Encode categorical features as a one-hot numeric array
2. Split to test and train series
3. Scale number
4. Select statistically relevant features
5. Train model and compute accuraccy score for
    1. Decision Tree 
    2. SVM 
    3. Kernel SVM
    4. K-NN 
    5. Logistic Regression
    6. Naive Bayes 
    7. Random Forest
6. Save best model