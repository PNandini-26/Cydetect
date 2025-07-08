from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must come BEFORE importing pyplot
import matplotlib.pyplot as plt
import io
import base64
import logging
import mysql.connector
from mysql.connector import Error
import os

# Get the root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, '..', 'templates'),
    static_folder=os.path.join(ROOT_DIR, '..', 'static')
)
app.secret_key = 'your_secret_key'


# Load models and vectorizers
sentiment_model = joblib.load(os.path.join(ROOT_DIR, '..', 'hack_model.pkl'))
sentiment_tfidf = joblib.load(os.path.join(ROOT_DIR, '..', 'hack_tfidf_vectorizer.pkl'))
cyberbullying_model = joblib.load(os.path.join(ROOT_DIR, '..', 'hack2_model.pkl'))
cyberbullying_tfidf = joblib.load(os.path.join(ROOT_DIR, '..', 'hack2_tfidf_vectorizer.pkl'))
label_encoder = joblib.load(os.path.join(ROOT_DIR, '..', 'label_encoder.pkl'))

# Load the trained models, TF-IDF vectorizers, and label encoders
sentiment_model = joblib.load(os.path.join(ROOT_DIR,  'hack_model.pkl'))
sentiment_tfidf = joblib.load(os.path.join(ROOT_DIR,  'hack_tfidf_vectorizer.pkl'))
cyberbullying_model = joblib.load(os.path.join(ROOT_DIR,  'hack2_model.pkl'))
cyberbullying_tfidf = joblib.load(os.path.join(ROOT_DIR,  'hack2_tfidf_vectorizer.pkl'))
label_encoder = joblib.load(os.path.join(ROOT_DIR,  'label_encoder.pkl'))


# MySQL Config
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'nand',
    'database': 'cydetect'
}

def init_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(50) NOT NULL,
                email VARCHAR(100) NOT NULL UNIQUE
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                action VARCHAR(100) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        conn.commit()
    except Error as e:
        print(f"Error: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    if 'user_id' not in session:
        flash('Please log in or register to access this feature.')
        return redirect(url_for('login'))
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        tweet_text = data.get('tweet_text', '')

        if not tweet_text.strip():
            return jsonify({'error': 'No tweet text provided.'})

        tfidf_input = cyberbullying_tfidf.transform([tweet_text])
        prediction_encoded = cyberbullying_model.predict(tfidf_input)
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]

        return jsonify({'prediction': prediction})

    except Exception as e:
        logging.error(f"Prediction Error: {e}")
        return jsonify({'error': str(e)})

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if 'user_id' not in session:
        flash('Please log in or register to access this feature.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'dataset_file' not in request.files:
            return jsonify({'error': 'No file part in the request'})

        file = request.files['dataset_file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        try:
            temp_path = os.path.join(ROOT_DIR, 'temp_dataset.csv')
            file.save(temp_path)
            df = pd.read_csv(temp_path)
            os.remove(temp_path)

            # Case 1: Sentiment Analysis
            if 'reviews.text' in df.columns and 'reviews.rating' in df.columns:
                df.dropna(subset=['reviews.text', 'reviews.rating'], inplace=True)
                df['sentiment'] = df['reviews.rating'].apply(lambda x: 'positive' if x > 3 else 'negative')
                X = df['reviews.text']
                X_tfidf = sentiment_tfidf.transform(X)
                preds = sentiment_model.predict(X_tfidf)
                pred_counts = pd.Series(preds).value_counts().to_dict()

                # Plotting chart
                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
                wedges, _, autotexts = ax.pie(
                    pred_counts.values(),
                    wedgeprops=dict(width=0.5),
                    startangle=-40,
                    autopct='%1.1f%%',
                    textprops=dict(color="w")
                )
                ax.set_title("Sentiment Analysis Results")
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close(fig)

                return jsonify({'type': 'sentiment', 'prediction_counts': pred_counts, 'plot_url': plot_url})

            # Case 2: Cyberbullying Detection
            elif 'tweet_text' in df.columns:
                df.dropna(subset=['tweet_text'], inplace=True)
                X = df['tweet_text']
                X_tfidf = cyberbullying_tfidf.transform(X)
                preds = cyberbullying_model.predict(X_tfidf)
                decoded = label_encoder.inverse_transform(preds)
                pred_counts = pd.Series(decoded).value_counts().to_dict()

                return jsonify({'type': 'cyberbullying', 'prediction_counts': pred_counts})

            else:
                return jsonify({'error': 'CSV must contain either reviews.text/reviews.rating or tweet_text'})

        except Exception as e:
            logging.error(f"Error during analysis: {e}")
            return jsonify({'error': str(e)})

    return render_template('analysis.html')
    if 'user_id' not in session:
        flash('Please log in or register to access this feature.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'dataset_file' not in request.files:
            return jsonify({'error': 'No file part in the request'})
        file = request.files['dataset_file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        try:
            temp_path = os.path.join(ROOT_DIR, 'temp_dataset.csv')
            file.save(temp_path)
            new_df = pd.read_csv(temp_path)

            if 'reviews.text' in new_df.columns and 'reviews.rating' in new_df.columns:
                new_df = new_df.dropna(subset=['reviews.text', 'reviews.rating'])
                new_df['sentiment'] = new_df['reviews.rating'].apply(lambda x: 'positive' if x > 3 else 'negative')
                new_X = new_df['reviews.text']
                new_X_tfidf = sentiment_tfidf.transform(new_X)
                new_predictions = sentiment_model.predict(new_X_tfidf)
                prediction_counts = pd.Series(new_predictions).value_counts().to_dict()

                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))
                wedges, _, autotexts = ax.pie(
                    prediction_counts.values(),
                    wedgeprops=dict(width=0.5),
                    startangle=-40,
                    autopct='%1.1f%%',
                    textprops=dict(color="w")
                )
                ax.set_title("Sentiment Analysis Results")

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close(fig)

                os.remove(temp_path)
                return jsonify({'prediction_counts': prediction_counts, 'plot_url': plot_url})

            elif 'tweet_text' in new_df.columns:
                new_df = new_df.dropna(subset=['tweet_text'])
                new_X = new_df['tweet_text']
                new_X_tfidf = cyberbullying_tfidf.transform(new_X)
                new_predictions = cyberbullying_model.predict(new_X_tfidf)
                decoded = label_encoder.inverse_transform(new_predictions)
                prediction_counts = pd.Series(decoded).value_counts().to_dict()

                os.remove(temp_path)
                return jsonify({'prediction_counts': prediction_counts})
            else:
                os.remove(temp_path)
                return jsonify({'error': 'Dataset is missing required columns'})

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return jsonify({'error': str(e)})

    return render_template('analysis.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            cursor.execute('SELECT id, password FROM users WHERE username = %s', (username,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user and user['password'] == password:
                session['user_id'] = user['id']
                flash('Login successful!')
                return redirect(url_for('home'))
            else:
                flash('Invalid username or password')
        except Error as e:
            flash(f'Error: {e}')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (username, password, email) VALUES (%s, %s, %s)',
                (username, password, email)
            )
            conn.commit()
            cursor.close()
            conn.close()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        except Error as e:
            flash(str(e))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.')
    return redirect(url_for('home'))

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please log in or register to access this feature.')
        return redirect(url_for('login'))
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT action, timestamp FROM history WHERE user_id = %s', (session['user_id'],))
        history = cursor.fetchall()
        cursor.close()
        conn.close()
        return render_template('histroy.html', history=history)
    except Error as e:
        flash(str(e))
        return render_template('histroy.html')

if __name__ == '__main__':
    app.run(debug=True)
