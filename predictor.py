import pickle

def predict(data):
    filename = 'logistic_regression.sav'

    print(data)

    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model.predict([data])

filename = 'logistic_regression.sav'

loaded_model = pickle.load(open(filename, 'rb'))

res = loaded_model.predict(["POPE FRANCIS IS GAY"])

print(res)
