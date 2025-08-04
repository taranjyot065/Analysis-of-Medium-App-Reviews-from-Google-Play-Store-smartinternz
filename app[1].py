from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
with open('disClassifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf.pkl', 'rb') as tfidf_file:
    vectorizer = pickle.load(tfidf_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['text']
        transformed_text = vectorizer.transform([input_text])
        prediction = model.predict(transformed_text)[0]
        return render_template('index.html', prediction=prediction, text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
