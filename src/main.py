from data.DataLoader import DataLoader
from data.DataProcess import DataPreparator
from model.model import *
from plots.render import *

data_loader = DataLoader(true_path="/home/henrique/personal_projects/NLP-fake-news-detection/data/True.csv",
                         fake_path="/home/henrique/personal_projects/NLP-fake-news-detection/data/Fake.csv")

data_preparator = DataPreparator()
model_training = Training('all')
model_testing = Testing()
model_prediction = Prediction()


true, fake = data_loader.load()

data = data_preparator.prepare(true, fake)

x_train,x_test,y_train,y_test = data_preparator.split_data(data=data)

model = model_training.fitting(x_train=x_train, y_train=y_train)

results = model_testing.choose_best_model(model=model, data=x_test)


save(model, 'logistic_regression.sav')

plot_feature(data, 'subject')
plot_feature(data, 'is_real')