import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pickle

model = pickle.load(open('model.pkl', 'rb'))
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        counts = Counter(y)
        impurity = 1.0
        for label in counts:
            prob_of_label = counts[label] / len(y)
            impurity -= prob_of_label**2
        return impurity

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_split = None
        n_features = X.shape[1]

        for feature_index in range(n_features):
            unique_values = np.unique(X[:, feature_index])
            if len(unique_values) <= 1:
                continue

            thresholds = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values) - 1)]

            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]

                y_left = y[left_indices]
                y_right = y[right_indices]

                weight_left = len(y_left) / len(y)
                weight_right = len(y_right) / len(y)

                gini = (weight_left * self._gini_impurity(y_left) +
                        weight_right * self._gini_impurity(y_right))

                if gini < best_gini:
                    best_gini = gini
                    best_split = {'feature_index': feature_index, 'threshold': threshold}

        return best_split

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return Node(value=Counter(y).most_common(1)[0][0])

        split = self._best_split(X, y)
        if split is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        feature_index = split['feature_index']
        threshold = split['threshold']

        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]

        left_node = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_index=feature_index, threshold=threshold, left=left_node, right=right_node)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _predict_single(self, x):
        node = self.root
        while node.value is None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        n_samples = X.shape[0]

        for i in range(self.n_estimators):
            print(f"Training tree {i+1} of {self.n_estimators}...")

            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_subset = X[indices]
            y_subset = y[indices]

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_subset, y_subset)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        final_predictions = np.array([Counter(predictions[:, i]).most_common(1)[0][0] for i in range(predictions.shape[1])])
        return final_predictions

def train_and_save_model():
    try:
        file_path = 'healthcare-dataset-stroke-data.csv'
        df = pd.read_csv(file_path)
        df = df.drop('id', axis=1)

        df['bmi'] = df['bmi'].replace('N/A', np.nan)
        df['bmi'] = pd.to_numeric(df['bmi'])
        df['bmi'] = df['bmi'].fillna(df['bmi'].median())
        df = df[df['gender'] != 'Other']

        categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        encoders = {}
        for feature in categorical_features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            encoders[feature] = le

        X = df.drop('stroke', axis=1).to_numpy()
        y = df['stroke'].to_numpy()

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        with open('stroke_model.pkl', 'wb') as model_file:
            pickle.dump(rf_model, model_file)

        with open('label_encoders.pkl', 'wb') as encoders_file:
            pickle.dump(encoders, encoders_file)

        print("Model and encoders saved successfully.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    train_and_save_model()


