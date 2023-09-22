import pickle

def predict(data):
    filename = 'logistic_regression.sav'

    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model.predict([data])
