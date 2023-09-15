from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import pickle


class Training:
    def __init__(self, model_name) -> None:
        self.model_list = {
            "Logistic Regression" : LogisticRegression(), 
            "Decision Tree" : DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42),
            "Random Forest" : RandomForestClassifier(n_estimators=50, criterion="entropy")
        }
        self.model_name = model_name

        if model_name == 'all':
            self.models = {}
            for name, model in self.model_list.items():
                self.models[name] = self.create_pipeline(model)
        elif model_name in self.model_list:
            self.model = self.create_pipeline(self.model_list[model_name])
        else:
            raise ValueError("Model isnt implemented")

    def create_pipeline(self, model):
        return Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('model', model)])

    def fitting(self, x_train, y_train):
        if self.model_name == 'all':
            for name, model in self.models.items():
                model.fit(x_train, y_train)
        else:
            self.model.fit(x_train, y_train)

class Testing:    

    def choose_best_model(self, models, x_test, y_test):
        best_model = None
        best_accuracy = 0.0

        for name, model in models.items():
            preds = model.predict(x_test)
            accuracy = accuracy_score(y_test, preds)
            print(f"{name} accuracy: {accuracy * 100:.2f}%")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = name

        print(f"The best model is: {best_model} with accuracy {best_accuracy * 100:.2f}%")

        return best_model

class Prediction(Testing):
    pass

def save(model, file_name):
    pickle.dump(model, open(file_name, 'wb'))




