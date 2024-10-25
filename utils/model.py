import numpy as np

class Model:
    def __init__(self, params={}):
        self.params = params

    def train(self, X, y):
        pass

    def predict(self, X):
        raise NotImplementedError

    def evaluate(self, X, y_labels):
        assert len(X) == len(y_labels), 'X and y must have the same length'
        y_preds = self.predict(X)
        return 1 - np.mean(abs(y_preds - y_labels))
    
    @staticmethod
    def process_sex(series, dict = {'male': -1, 'female': 1}):
        return series.map(dict).astype(int)

        

class ConstantBaselineModel(Model):
    def __init__(self, params={'survived': False}):
        super().__init__(params)

    def predict(self, X):
        return np.zeros(X.shape[0]) if not self.params['survived'] else np.ones(X.shape[0])
    

class SexBaselineModel(Model):
    def __init__(self, params={}):
        super().__init__(params)

    def predict(self, X):
        sex = self.process_sex(X['Sex'])
        return ((sex + 1) / 2).astype(int)
    
