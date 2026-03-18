from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

app = Flask(__name__)

# Load dataset
data = pd.read_csv("auto_mpg.csv")

# ---------------- DATA CLEANING ----------------
data = data.replace('?', np.nan)
data = data.dropna()

# Remove non-useful column if exists
if 'car name' in data.columns:
    data = data.drop('car name', axis=1)

# Convert horsepower to float
data['horsepower'] = data['horsepower'].astype(float)

# ---------------- FEATURES ----------------
X = data[['cylinders', 'horsepower', 'weight', 'acceleration']]
y = data['mpg']

# ---------------- SCALING (better accuracy) ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- TRAIN TEST ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy check
preds = model.predict(X_test)
print("Model Accuracy (R2 Score):", r2_score(y_test, preds))

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        cyl = float(request.form['cylinders'])
        hp = float(request.form['horsepower'])
        wt = float(request.form['weight'])
        acc = float(request.form['acceleration'])

        input_data = scaler.transform([[cyl, hp, wt, acc]])
        prediction = model.predict(input_data)[0]

        return render_template(
            'index.html',
            prediction=round(prediction, 2),
            cyl=cyl, hp=hp, wt=wt, acc=acc
        )

    except:
        return render_template('index.html', prediction="Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)