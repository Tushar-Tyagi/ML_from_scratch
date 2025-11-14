import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier as DTClassifier



class Bagging():
    def __init__(self,n_models=100):
        self.n_models=n_models
        self.random_states = [i for i in range(n_models)]
        self.tree_models = []

    def bootstapping(self,X,y):
        n_samples = X.shape[0]
        #generate n_samples in range [0,n_samples)
        idxs = np.random.choice(n_samples,n_samples,replace=True)
        return X[idxs],y[idxs]

    def fit(self, X, y):
        for i in range(self.n_models):
            X_,y_ = self.bootstapping(X,y)
            tree = DTClassifier(max_depth=2,random_state=self.random_states[i])
            tree.fit(X_,y_)
            self.tree_models.append(tree)
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.tree_models])
        predictions = stats.mode(predictions)
        return predictions[0]










