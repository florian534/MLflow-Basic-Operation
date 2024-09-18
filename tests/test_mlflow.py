import pytest
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import sys
import os 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from example import eval_metrics



# 1er exemple test unitaire : Test du modèle ElasticNet pour s'assurer qu'il peut s'entraîner
def test_model_training():
    data_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    data = pd.read_csv(data_url, sep=";")
    train = data.sample(frac=0.75, random_state=42)
    train_x = train.drop(["quality"], axis=1)
    train_y = train[["quality"]]

    model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
    model.fit(train_x, train_y)

    predictions = model.predict(train_x)
    mse = mean_squared_error(train_y, predictions)

    # Vérification que le MSE est inférieur à une certaine valeur
    assert mse < 0.9


#Vérifie que la fonction eval_metrics() renvoie des valeurs cohérentes pour le RMSE, le MAE et le R2 en utilisant des valeurs prédéfinies.
#Test du modèle test_model_training() :

#Entraîne un modèle ElasticNet sur les données de vin, puis vérifie que le Mean Squared Error est inférieur à 0.5.



#--------------------------------------------------------
#Exemple de test unitaire pour la fonction eval_metrics
#def test_eval_metrics():
    #actual = [3, 5, 7, 9]
    #pred = [2.5, 5.0, 7.5, 9.0]
    #rmse, mae, r2 = eval_metrics(actual, pred)

    #assert pytest.approx(rmse, 0.1) == 0.5 #pytest.approx(rmse, 0.1) : Cette fonction permet de vérifier que la valeur de rmse (Root Mean Squared Error) est proche de 0.5 avec une tolérance de 0.1. Cela signifie que rmse peut être dans l'intervalle [0.4, 0.6].
    #assert pytest.approx(mae, 0.1) == 0.5
    #assert pytest.approx(r2, 0.1) == 0.5
#Explication des tests
#Test unitaire test_eval_metrics() :