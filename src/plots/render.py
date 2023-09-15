import matplotlib.pyplot as plt

def plot_feature(data, feature):
    data.groupby([feature])['text'].count().plot(kind="bar")
    plt.show()
