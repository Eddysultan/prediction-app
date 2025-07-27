from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# ✅ Create Flask app instance (needed for Render/Gunicorn)
app = Flask(__name__)

# ✅ Load dataset
df = pd.read_csv('dataset.csv')

# ✅ Clean and convert nutrient columns
df['Vitamin_A_mg'] = pd.to_numeric(df['Vitamin_A_mg'], errors='coerce')
df['Vitamin_C_mg'] = pd.to_numeric(df['Vitamin_C_mg'], errors='coerce')
df['Calcium_mg'] = pd.to_numeric(df['Calcium_mg'], errors='coerce')
df['Iron_mg'] = pd.to_numeric(df['Iron_mg'], errors='coerce')
df = df.dropna(subset=['Vitamin_A_mg', 'Vitamin_C_mg', 'Calcium_mg', 'Iron_mg'])

# ✅ Create binary deficiency labels
df['Iron_Def'] = df.apply(lambda row: row['Iron_mg'] < 8 if row['Gender'] == 'Female' else row['Iron_mg'] < 10, axis=1)
df['Vitamin_A_Def'] = df['Vitamin_A_mg'] < 0.9
df['Vitamin_C_Def'] = df['Vitamin_C_mg'] < 75
df['Calcium_Def'] = df['Calcium_mg'] < 1000

# ✅ Feature engineering
df['VitA_density'] = df['Vitamin_A_mg'] / (df['Daily_Calorie_Intake'] / 1000)
df['VitC_density'] = df['Vitamin_C_mg'] / (df['Daily_Calorie_Intake'] / 1000)
df['Calcium_density'] = df['Calcium_mg'] / (df['Daily_Calorie_Intake'] / 1000)
df['Iron_density'] = df['Iron_mg'] / (df['Daily_Calorie_Intake'] / 1000)
df['Iron_to_VitC'] = df['Iron_mg'] / (df['Vitamin_C_mg'] + 1)
df['Calcium_to_VitC'] = df['Calcium_mg'] / (df['Vitamin_C_mg'] + 1)

# ✅ Socioeconomic tiers
occupation_tiers = {
    'Doctor': 'High', 'Engineer': 'High', 'Software Developer': 'High', 'Business Woman': 'Medium',
    'Accountant': 'Medium', 'Teacher': 'Medium', 'Marketer': 'Medium', 'Farmer': 'Low', 'Driver': 'Low',
    'Builder': 'Low', 'Self Employed': 'Low', 'Nurse': 'Medium', 'Banker': 'High', 'Actor': 'Medium',
    'Actress': 'Medium', 'Student': 'Low', 'Drummer': 'Low'
}
education_tiers = {
    'PhD': 'High', 'Master': 'High', 'Bachelor': 'Medium', 'High School': 'Low'
}
df['Occupation_Tier'] = df['Occupation'].map(occupation_tiers)
df['Education_Tier'] = df['Education_Level'].map(education_tiers)

# ✅ Energy expenditure mapping
activity_energy_map = {'Low': 200, 'Moderate': 350, 'High': 500}
df['Energy_Expenditure'] = df['Physical_Activity_Level'].map(activity_energy_map)

# ✅ Prepare data
target_cols = ['Vitamin_A_Def', 'Vitamin_C_Def', 'Calcium_Def', 'Iron_Def']
y = df[target_cols].astype(int)
X = df.drop(columns=target_cols + ['Vitamin_A_mg', 'Vitamin_C_mg', 'Calcium_mg', 'Iron_mg', 'Diet_Score'])

# ✅ Encode categorical features
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# ✅ Impute and scale features
X_imputed = SimpleImputer(strategy='median').fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_imputed)

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Train model with class weights
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model = MultiOutputClassifier(rf)
model.fit(X_train, y_train)

# ✅ Print evaluation in logs (optional)
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=target_cols)
print(report)

# ✅ Home route (to test Render deployment)
@app.route('/')
def home():
    return "✅ Prediction API is running on Render!"

# ✅ Prediction route (POST)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Convert JSON input to DataFrame
        input_df = pd.DataFrame([data])

        # Encode categorical variables like we did for training
        for col in input_df.select_dtypes(include='object').columns:
            input_df[col] = input_df[col].astype(str)

        # Match training columns
        missing_cols = set(X.columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0  # add missing columns with default 0

        # Reorder columns to match training
        input_df = input_df[X.columns]

        # Impute & scale
        input_imputed = SimpleImputer(strategy='median').fit(X).transform(input_df)
        input_scaled = StandardScaler().fit(X).transform(input_imputed)

        # Predict
        prediction = model.predict(input_scaled)

        # Build response
        response = {
            "Vitamin_A_Def": bool(prediction[0][0]),
            "Vitamin_C_Def": bool(prediction[0][1]),
            "Calcium_Def": bool(prediction[0][2]),
            "Iron_Def": bool(prediction[0][3])
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ✅ Required for local testing (Render ignores this and uses Gunicorn)
if __name__ == '__main__':
    app.run(debug=True)
