import pandas as pd

class DataLoader:
    def __init__(self, true_path, fake_path) -> None:
        self.true_path = true_path
        self.fake_path = fake_path  
    
    def load(self):
        true = pd.read_csv(self.true_path)
        fake = pd.read_csv(self.fake_path)

        return true, fake
    
