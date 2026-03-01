import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        """
        Initialize the KNN classifier.
        :param k: The number of nearest neighbors to consider.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data.
        KNN is a lazy learner, so we don't actually 'train' a model here.
        
        :param X: Training features (numpy array of shape [n_samples, n_features])
        :param y: Training labels (numpy array of shape [n_samples])
        """
        # TODO: Store X and y in self.X_train and self.y_train
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        """
        Calculate the Euclidean distance between two data points.
        
        :param x1: First data point (numpy array)
        :param x2: Second data point (numpy array)
        :return: Scalar distance
        """
        # TODO: Implement the distance formula: sqrt(sum((x1 - x2)^2))
        return np.sqrt(np.sum((x1-x2)**2))

    def _predict_one(self, x):
        """
        Predict the class label for a single example x.
        
        :param x: A single data point (numpy array)
        :return: Predicted class label
        """
        # 1. Calculate distances between x and all examples in self.X_train.
        
        # 2. Sort the distances and get the indices of the first k neighbors.
        
        # 3. Extract the labels of those k nearest neighbors.
        
        # 4. Return the most common label (majority vote).
        distances = np.array([self._euclidean_distance(x, x_train) for x_train in self.X_train])
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = self.y_train[nearest_indices]
        most_common = Counter(nearest_labels).most_common(1)
        # most common returns a list of tuples (label, count), we want the label
        return most_common[0][0]

    def predict(self, X):
        """
        Predict class labels for multiple examples.
        
        :param X: Test features (numpy array of shape [n_samples, n_features])
        :return: Array of predicted labels
        """
        # TODO: Apply _predict_one to every row in X
        # Hint: You can use a list comprehension or np.apply_along_axis
        return np.array([self._predict_one(x) for x in X])

# --- Testing Setup (Do not modify this part initially) ---
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize your model
    clf = KNNClassifier(k=3)
    
    # Fit
    clf.fit(X_train, y_train)
    
    # Predict
    predictions = clf.predict(X_test)
    
    # Evaluate
    print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")