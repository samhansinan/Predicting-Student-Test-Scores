from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and columns
model = joblib.load("student_score_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Function to generate suggestions based on input data
def generate_suggestions(input_data):
    suggestions = []

    if input_data['study_hours'] < 5:
        suggestions.append("Increase study hours to improve score.")
    
    if input_data['class_attendance'] < 70:
        suggestions.append("Attendance is low, consider improving it.")
    
    if input_data['sleep_hours'] < 6:
        suggestions.append("Try to get enough sleep for better concentration.")
    
    if input_data.get('exam_difficulty', '').lower() == 'hard':
        suggestions.append("Prepare more thoroughly for difficult exams.")
    
    if input_data.get('facility_rating', '').lower() == 'low':
        suggestions.append("Improve access to study resources if possible.")

    if not suggestions:
        suggestions.append("Keep up the good work!")

    return suggestions

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form data from user
        data = request.form.to_dict()

        # Convert numeric fields in the dictionary too
        numeric_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
        for col in numeric_cols:
            data[col] = float(data[col])  # convert to float

        # Convert to DataFrame for model prediction
        df = pd.DataFrame([data])

        # One-hot encode categorical features
        categorical_cols = ['gender', 'course', 'internet_access', 
                            'sleep_quality', 'study_method', 
                            'facility_rating', 'exam_difficulty']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Align columns with model's training data
        df = df.reindex(columns=model_columns, fill_value=0)

        # Predict exam score
        prediction = model.predict(df)[0]

        # Generate suggestions (now numeric fields are floats)
        suggestions = generate_suggestions(data)

        # Render template with prediction and suggestions
        return render_template('index.html', 
                               prediction=round(prediction, 2),
                               suggestions=suggestions)

    # GET request just shows the form
    return render_template('index.html', prediction=None, suggestions=None)


if __name__ == '__main__':
    app.run(debug=True)
