
from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threading import Thread

# Inicializar o aplicativo Flask
app = Flask(__name__)

# Carregar o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar e treinar o modelo
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


@app.route('/')
def home():
    return "API de Classificação do Iris Dataset"

# Rota para fazer previsões
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Obter os parâmetros de entrada da requisição
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))

        # Carregar o modelo e o escalador
        model = joblib.load('iris_model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Normalizar os dados de entrada
        input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])

        # Fazer a previsão
        prediction = model.predict(input_data)

        # Mapear a previsão para o nome da classe
        iris_target_names = iris.target_names
        predicted_class = iris_target_names[prediction[0]]

        # Retornar a previsão como resposta JSON
        return jsonify({
            'predicted_class': predicted_class
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Function to run the Flask app
def run():
    app.run(port=5000, debug=False, use_reloader=False)  # Prevents reloading, works better in notebooks

# Start the Flask app in a background thread
thread = Thread(target=run)
thread.start()
