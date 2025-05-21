from flask import Flask, request, render_template_string
import pandas as pd
import pickle
import os

app = Flask(__name__)

# File paths
MODEL_FILE = 'rf_model.pkl'
DATA_FILE = 'rrrr.csv'

# Column definitions based on the image
categorical_columns = [
    'age_band (to number)',
    'disability (to number)'
]
binary_fields = ['disability (to number)']
numerical_columns = [
    'num_of_prev_attempts',
    'studied_credits',
    'final_score',
    'sum_click',
    'count_click'
]
features = numerical_columns + categorical_columns

# Load or train the model
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        return model

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE

    # Load and preprocess data
    data = pd.read_csv(DATA_FILE)
    data['final_result'] = data['final_result'].apply(lambda x: 1 if x == 'Withdrawn' else 0)
    X = data[features]
    y = data['final_result']

    # Handle missing values
    X[numerical_columns] = X[numerical_columns].apply(pd.to_numeric, errors='coerce')
    X[numerical_columns] = X[numerical_columns].fillna(X[numerical_columns].mean())
    X[categorical_columns] = X[categorical_columns].fillna(X[categorical_columns].mode().iloc[0])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Save the model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    return model

model = load_or_train_model()

# Function to generate recommendations based on input data
def generate_recommendations(input_data):
    recommendations = []
    
    # Convert input data to appropriate types
    num_attempts = float(input_data.get('num_of_prev_attempts', 0))
    studied_credits = float(input_data.get('studied_credits', 0))
    final_score = float(input_data.get('final_score', 0))
    sum_click = float(input_data.get('sum_click', 0))
    count_click = float(input_data.get('count_click', 0))
    disability = int(input_data.get('disability (to number)', 0))
    
    # Recommendation rules
    if num_attempts > 2:
        recommendations.append("Consider academic counseling to address repeated course attempts. A tutor or study group may help improve understanding.")
    if studied_credits > 120:
        recommendations.append("High credit load detected. Consider reducing course load or seeking time management workshops to balance studies.")
    if final_score < 50:
        recommendations.append("Low final score indicates academic struggle. Engage with supplemental instruction or office hours for additional support.")
    if sum_click < 100 or count_click < 50:
        recommendations.append("Low engagement with the learning system. Increase interaction with course materials and participate in online discussions.")
    if disability == 1:
        recommendations.append("Ensure accessibility accommodations are in place. Contact the disability services office for tailored support.")
    
    return recommendations if recommendations else ["No specific recommendations. Continue monitoring progress and seek general academic support if needed."]

# HTML Template with added recommendations section
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Student Dropout Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f9;
        }
        h2 {
            color: #333;
            text-align: center;
        }
        form {
            background: #fff;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin: 15px 0 5px;
            font-weight: bold;
            color: #444;
        }
        input, select {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-sizing: border-box;
            font-size: 14px;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 5px rgba(0,123,255,0.3);
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            display: block;
            width: 100%;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .prediction {
            text-align: center;
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            background: #e9ecef;
        }
        .recommendations {
            text-align: left;
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            background: #fff3cd;
            border: 1px solid #ffeeba;
        }
        .recommendations ul {
            margin: 0;
            padding-left: 20px;
        }
        .error {
            color: #d9534f;
            text-align: center;
            margin-top: 10px;
            padding: 10px;
            background: #f8d7da;
            border-radius: 5px;
        }
        .description {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
            display: block;
        }
    </style>
    <script>
        function validateForm() {
            let valid = true;
            let errorMsg = '';
            
            // Validate numerical fields based on dataset ranges
            const numFields = {
                'num_of_prev_attempts': {min: 0, max: 10, desc: 'Number of Previous Attempts must be between 0 and 10'},
                'studied_credits': {min: 0, max: 300, desc: 'Studied Credits must be between 0 and 300'},
                'final_score': {min: 0, max: 100, desc: 'Final Score must be between 0 and 100'},
                'sum_click': {min: 0, max: 5000, desc: 'Sum Click must be between 0 and 5000'},
                'count_click': {min: 0, max: 1500, desc: 'Count Click must be between 0 and 1500'}
            };

            for (let field in numFields) {
                let value = document.forms["predictionForm"][field].value;
                let numValue = parseFloat(value);
                if (isNaN(numValue) || numValue < numFields[field].min || numValue > numFields[field].max) {
                    errorMsg += `Error: ${numFields[field].desc}.\n`;
                    valid = false;
                }
            }

            if (!valid) {
                document.getElementById('error-message').innerText = errorMsg;
                document.getElementById('error-message').style.display = 'block';
            } else {
                document.getElementById('error-message').style.display = 'none';
            }
            return valid;
        }
    </script>
</head>
<body>
    <h2>Student Dropout Prediction</h2>
    <form name="predictionForm" method="POST" onsubmit="return validateForm()">
        {% for col in numerical_columns %}
            <label>{{ col.replace('_', ' ').title() }}:</label>
            <div class="description">
                {% if col == 'num_of_prev_attempts' %}
                    Number of times the student attempted the course before. Range: 0 to 10.
                {% elif col == 'studied_credits' %}
                    Total credits the student is enrolled in. Range: 0 to 300.
                {% elif col == 'final_score' %}
                    Final score or grade (percentage). Range: 0 to 100.
                {% elif col == 'sum_click' %}
                    Total number of clicks in the learning system. Range: 0 to 5000.
                {% elif col == 'count_click' %}
                    Number of click events recorded. Range: 0 to 1500.
                {% endif %}
            </div>
            <input name="{{ col }}" type="number" step="any" 
                {% if col == 'num_of_prev_attempts' %} min="0" max="10"
                {% elif col == 'studied_credits' %} min="0" max="300"
                {% elif col == 'final_score' %} min="0" max="100"
                {% elif col == 'sum_click' %} min="0" max="5000"
                {% elif col == 'count_click' %} min="0" max="1500"
                {% endif %}
                required><br>
        {% endfor %}
        {% for col in categorical_columns %}
            <label>{{ col.replace('(to number)', '').replace('_', ' ').title() }}:</label>
            <div class="description">
                {% if col == 'age_band (to number)' %}
                    Age range of the student. Options: 0-35 (0), 35-55 (1), 55<= (2).
                {% elif col == 'disability (to number)' %}
                    Whether the student has a disability. Options: No (0), Yes (1).
                {% endif %}
            </div>
            {% if col == 'age_band (to number)' %}
                <select name="{{ col }}" required>
                    <option value="0">0-35</option>
                    <option value="1">35-55</option>
                    <option value="2">55<=</option>
                </select>
            {% elif col == 'disability (to number)' %}
                <select name="{{ col }}" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            {% endif %}
            <br>
        {% endfor %}
        <button type="submit">Predict</button>
        <div id="error-message" class="error" style="display: none;"></div>
    </form>

    {% if prediction is not none %}
        <div class="prediction">
            <h3>Prediction: <span style="color: {{ 'red' if prediction == 'Dropout' else 'green' }}">{{ prediction }}</span></h3>
            <p>Probability of Dropout: <span style="font-weight: bold;">{{ (probability * 100)|round(2) }}%</span></p>
        </div>
        {% if prediction == 'Dropout' and recommendations %}
            <div class="recommendations">
                <h3>Recommended Actions to Prevent Dropout:</h3>
                <ul>
                    {% for rec in recommendations %}
                        <li>{{ rec }}</li>
                    {% endfor %}
                </ul>
            </div>
        {% endif %}
    {% endif %}
</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    recommendations = None
    if request.method == "POST":
        try:
            input_data = {col: request.form.get(col, '') for col in features}
            df = pd.DataFrame([input_data])

            # Convert numericals to float
            for col in numerical_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert categoricals to int
            for col in categorical_columns:
                df[col] = df[col].astype(int)

            # Handle missing values
            df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
            df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

            # Predict
            pred = model.predict(df[features])[0]
            prob = model.predict_proba(df[features])[0][1]
            prediction = "Dropout" if pred == 1 else "No Dropout"
            probability = prob

            # Generate recommendations if prediction is Dropout
            if prediction == "Dropout":
                recommendations = generate_recommendations(input_data)
                
        except ValueError as ve:
            prediction = f"Error: Invalid input value. Ensure all fields are numeric and within the specified ranges. Details: {str(ve)}"
        except Exception as e:
            prediction = f"Error: An unexpected issue occurred. Details: {str(e)}"

    return render_template_string(HTML_TEMPLATE,
                                 prediction=prediction,
                                 probability=probability,
                                 recommendations=recommendations,
                                 categorical_columns=categorical_columns,
                                 numerical_columns=numerical_columns,
                                 binary_fields=binary_fields)

if __name__ == "__main__":
    app.run(debug=True)