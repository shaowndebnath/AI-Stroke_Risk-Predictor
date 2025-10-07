from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import requests
import json
import os
from sklearn.ensemble import RandomForestClassifier
from train_model import RandomForestClassifier
import pickle
model = pickle.load(open('model.pkl', 'rb'))
from collections import Counter

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
    
    def predict_proba(self, X):
        print("Warning: predict_proba method is a placeholder and may not give correct results.")
        predictions = np.array([tree.predict(X) for tree in self.trees])
        probabilities = []
        for i in range(predictions.shape[1]):
            counts = Counter(predictions[:, i])
            total = len(predictions[:, i])
            prob_class_0 = counts.get(0, 0) / total
            prob_class_1 = counts.get(1, 0) / total
            probabilities.append([prob_class_0, prob_class_1])
        return np.array(probabilities)

app = Flask(__name__)

try:
    with open("stroke_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
except FileNotFoundError:
    print("Error: Model and encoder files not found. Please ensure 'stroke_model.pkl' and 'label_encoders.pkl' are in the same directory.")
    model = None
    encoders = None

GEMINI_API_KEY = "AIzaSyC5QBVqPD8dHZ8690SJf6IjJQtILQBPdck"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

def get_gemini_suggestion(prompt_text):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {
            "parts": [
                {
                    "text": "You are a health assistant providing general, non-medical advice. Offer actionable, positive, and simple suggestions for a healthier lifestyle based on the user's information. Do not diagnose or recommend specific treatments. Always advise consulting a healthcare professional for personalized advice."
                }
            ]
        },
    }
    try:
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        generated_text = result.get('candidates', [])[0].get('content', {}).get('parts', [])[0].get('text', 'No suggestion generated.')
        return generated_text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Error getting suggestion"

@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/verification.html", methods=["GET", "POST"])
def verification():
    if request.method == "POST":
        return redirect(url_for("main"))
    return render_template("verification.html")

@app.route("/main.html", methods=["GET", "POST"])
def main():
    prediction = None
    suggestion = None
    probability = None
    
    if request.method == "POST":
        if model is None or encoders is None:
            prediction = "Error: ML model is not loaded."
            suggestion = "Please check the server logs for missing files."
            return render_template("main.html", prediction=prediction, suggestion=suggestion, probability=probability)
        
        try:
            gender = int(request.form["gender"])
            age = float(request.form["age"])
            hypertension = int(request.form["hypertension"])
            heart_disease = int(request.form["heart_disease"])
            ever_married = int(request.form["ever_married"])
            work_type = int(request.form["work_type"])
            residence = request.form["residence"]
            glucose = float(request.form["glucose"])
            bmi = float(request.form["bmi"])
            smoking = int(request.form["smoking"])
            
            residence_encoded = encoders["Residence_type"].transform([residence])[0]
            X_input = np.array([[gender, age, hypertension, heart_disease, ever_married,
                                 work_type, residence_encoded, glucose, bmi, smoking]])
            
            pred = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0][1]
            
            if pred == 1:
                prediction_text = "High risk of Stroke!"
                prompt = f"The patient has high stroke risk. Age: {age}, Hypertension: {hypertension}, Heart: {heart_disease}, Glucose: {glucose}, BMI: {bmi}, Smoking: {smoking}."
            else:
                prediction_text = "Low risk of Stroke"
                prompt = f"The patient has low stroke risk. Age: {age}, Hypertension: {hypertension}, Heart: {heart_disease}, Glucose: {glucose}, BMI: {bmi}, Smoking: {smoking}."
                
            suggestion = get_gemini_suggestion(prompt)
            prediction = prediction_text
            
        except Exception as e:
            prediction = f"Error: {e}"
            suggestion = "Check your input values."
    
    return render_template("main.html", prediction=prediction, suggestion=suggestion, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)


