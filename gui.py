from flask import Flask, request
from flask import render_template
from keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model('trained_model.hdf5',compile=False)

@app.route('/')
def hello_world():

     return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    predict_data = []
    for parameter in request.args:
        index = int(str(parameter).split('_')[1])
        data = int(request.args.get(parameter))
        predict_data.insert(index,data)
    predict_array=np.array(predict_data).reshape(1, 9)
    predict = model.predict_classes(predict_array)
    if predict==2:
        return 'Well-behaved cells were detected.'
    else:
        return 'Malignant cancer cells were detected.'


if __name__ == '__main__':
    app.run()
