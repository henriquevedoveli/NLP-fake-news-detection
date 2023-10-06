import pandas as pd
from sklearn.utils import shuffle
import nltk
from nltk.corpus import stopwords
import string

from sklearn.model_selection import train_test_split

class DataPreparator():
    nltk.download('stopwords')

    @staticmethod
    def join(true, fake):
        return pd.concat([fake, true]).reset_index(drop = True)
    
    def prepare(self, true, fake):
        true['is_real'] = 1
        fake['is_real'] = 0

        data = self.join(true = true, fake=fake)
        data = shuffle(data)
        data = data.reset_index(drop=True)

        data['text'] = data['text'].apply(self.punctuation_removal)
        data['title'] = data['title'].apply(self.punctuation_removal)

        data['text'] = self.lowercase(data['text'])
        data['title'] = self.lowercase(data['title'])

        # data['text'] = self.remove_stop_words(data['text'])
        # data['title'] = self.remove_stop_words(data['title'])

        return data
    
    def punctuation_removal(self, text):
        all_list = [char for char in text if char not in string.punctuation]
        clean_str = ''.join(all_list)
        return clean_str
    
    @staticmethod
    def lowercase(text):
        return text.apply(lambda x: x.lower())
    
    @staticmethod
    def generate_stop_words():
        return stopwords.words('english')
    
    def remove_stop_words(self, text):
        print('This process may take sometime...')
        stop = self.generate_stop_words()

        return text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    @staticmethod
    def split_data(data, test_size=0.2):
        x_train,x_test,y_train,y_test = train_test_split(data.text, data.is_real, test_size=test_size, random_state=42)
        return x_train,x_test,y_train,y_test



