import pickle
from flask import Flask, request
from perceptron import Perceptron
from flask_restful import Api


with open("RTA_model_pick.pkl", 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
api = Api(app)


@app.route('/')
def start():
    return 'Witam'

#/api/predict?sl=coś&pl=coś
@app.route('/api/predict/', methods=['GET'])
def home():
    sl = request.args.get("sl", "4.5")
    pl = request.args.get("pl", "3.2")
    res = model.predict([float(sl), float(pl)])
    mapper = {'0': 'setosa',
              '1': 'versicolor'}
    return mapper[f"{res}"]

app.run(port='5032',host='0.0.0.0')




