from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# Load model, scaler, and encoders
def load_pickle(filename):
    with open(os.path.join('model', filename), 'rb') as f:
        return pickle.load(f)

model = xgb.XGBClassifier()
model.load_model(os.path.join('model', 'stroke_model.json'))

scaler = load_pickle('scaler.pkl')
label_encoders = load_pickle('label_encoders.pkl')

# Features used in training after feature selection
FEATURE_NAMES = [
    'gender', 'age', 'ever_married', 'work_type',
    'Residence_type', 'avg_glucose_level'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup')
def signup():
    if request.method == 'POST':
        name = request.form['name']
        gender = request.form['gender']
        contact = request.form['contact']
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirm']

        if password != confirm:
            flash("Passwords don't match!")
            return redirect('/signup')

        try:
            with sqlite3.connect("users.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO users (name, gender, contact, email, password) VALUES (?, ?, ?, ?, ?)",
                            (name, gender, contact, email, password))
                con.commit()
                flash("Account created successfully!")
                return redirect('/login')
        except:
            flash("Email already registered!")
            return redirect('/signup')
    return render_template('signup.html')

@app.route('/login')
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        con = sqlite3.connect("users.db")
        cur = con.cursor()
        cur.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
        user = cur.fetchone()
        con.close()

        if user:
            session['user'] = user[1]  # Storing name
            return f"Welcome, {user[1]}!"
        else:
            flash("Invalid credentials!")
            return redirect('/login')
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        form_data = {
            'gender': request.form['gender'],
            'age': float(request.form['age']),
            'ever_married': request.form['ever_married'],
            'work_type': request.form['work_type'],
            'Residence_type': request.form['residence_type'],
            'avg_glucose_level': float(request.form['avg_glucose_level']),
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([form_data])

        # Apply saved LabelEncoders
        for col in input_df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                if input_df[col][0] not in le.classes_:
                    # Handle unseen label by adding it temporarily
                    le.classes_ = np.append(le.classes_, input_df[col][0])
                input_df[col] = le.transform(input_df[col])

        # Reorder columns to match model input
        input_df = input_df[FEATURE_NAMES]

        # Scale the features
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        return render_template(
            'result.html',
            prediction=prediction,
            probability=round(probability * 100, 2)
        )

    except Exception as e:
        return f"Error occurred: {str(e)}", 400
    
@app.route('/doctor')
def doctor():
    return render_template('doctor.html')

@app.route('/about')
def about():
    team_members = [
        {'name': 'Dhanushikaa R', 'email': 'dhanushikaa7@gmail.com', 'university': 'Sastra University, Thanjavur'},
        {'name': 'Prathapaneni Kavya', 'email': 'kaviya2004@gmail.com', 'university': 'Sastra University, Thanjavur'},
        {'name': 'Swetha S', 'email': 'swetha19@gmail.com', 'university': 'Sastra University, Thanjavur'}
    ]
    return render_template('about.html', team_members=team_members)

if __name__ == '__main__':
    app.run(debug=True)

