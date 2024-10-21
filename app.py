from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import traceback

app = Flask(__name__)

# Charger le meilleur modèle
model_path = './models/elasticnet_model.joblib'  # Assurez-vous que le chemin est correct
model = joblib.load(model_path)

# Route pour la page web
@app.route('/')
def index():
    return render_template('index.html')

# Définir une route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    # Vérifier si la requête contient des données JSON
    data = request.get_json()
    print("Données reçues :", data)  # Afficher les données reçues dans la console

    # Vérifier que toutes les variables nécessaires sont présentes
    required_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                         'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                         'pH', 'sulphates', 'alcohol']
    if not all(feature in data for feature in required_features):
        return jsonify({'error': 'Missing required features'}), 400

    # Préparer les données pour la prédiction
    features = np.array([[data['fixed acidity'], data['volatile acidity'], data['citric acid'],
                          data['residual sugar'], data['chlorides'], data['free sulfur dioxide'],
                          data['total sulfur dioxide'], data['density'], data['pH'], data['sulphates'],
                          data['alcohol']]])

    # Faire la prédiction
    try:
        prediction = model.predict(features)
        print("Prédiction :", prediction[0])  # Afficher la prédiction dans la console
        prediction_value = int(prediction[0])  # Convertir numpy.int64 en type natif int
        return jsonify({'prediction': prediction_value})  # Retourner la prédiction
    except Exception as e:
        print("Erreur lors de la prédiction :", str(e))
        print(traceback.format_exc())  # Afficher les détails complets de l'erreur
        return jsonify({'error': 'Erreur lors de la prédiction : ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
