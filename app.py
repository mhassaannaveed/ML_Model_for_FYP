from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

# Load the trained models
model_files = {
    'diet_recommendation': 'diet_recommendation_model.pkl',
    'exercise_recommendation': 'exercise_recommendation_model.pkl',
    'bmr': 'bmr_model.pkl',
    'calories': 'calories_model.pkl'
}

models = {}
for model_name, file_name in model_files.items():
    with open(file_name, 'rb') as file:
        models[model_name] = pickle.load(file)

# Load the encoders from the file
with open('encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

    
gender_encoder = encoders['Gender']
fitness_level_encoder = encoders['Fitness Level']
medical_history_encoder = encoders['Medical History']
diet_recommendation_encoder = encoders['Diet Recommended']
exercise_recommendation_encoder = encoders['Exercise']

# Initialize the FastAPI app
app = FastAPI()

# Define the request body schema
class PredictionRequest(BaseModel):
    Age: list[int]
    Height: list[float]
    Weight: list[float]
    Gender: list[str]
    BMI: list[float]
    Fitness_Level: list[str]
    Medical_History: list[str]

# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Prepare the input data for prediction
        input_data = pd.DataFrame({
            'Age': request.Age,
            'Height': request.Height,
            'Weight': request.Weight,
            'Gender': request.Gender,
            'BMI': request.BMI,
            'Fitness Level': request.Fitness_Level,
            'Medical History': request.Medical_History
        })

        # Encode categorical features
        input_data['Gender'] = gender_encoder.transform(input_data['Gender'])
        input_data['Fitness Level'] = fitness_level_encoder.transform(input_data['Fitness Level'])
        input_data['Medical History'] = medical_history_encoder.transform(input_data['Medical History'])

        # Make predictions using the loaded models
        predictions = {}
        predictions['diet_recommendation'] = models['diet_recommendation'].predict(input_data)
        predictions['exercise_recommendation'] = models['exercise_recommendation'].predict(input_data)
        predictions['bmr'] = models['bmr'].predict(input_data)
        predictions['calories'] = models['calories'].predict(input_data)

        # Convert numeric predictions to categorical labels
        diet_recommendations = diet_recommendation_encoder.inverse_transform(predictions['diet_recommendation'].astype(int))
        exercise_recommendations = exercise_recommendation_encoder.inverse_transform(predictions['exercise_recommendation'].astype(int))

        # Return the predictions
        return {
            "Prediction for Diet Recommended": diet_recommendations.tolist(),
            "Prediction for Exercise": exercise_recommendations.tolist(),
            "Prediction for BMR": predictions['bmr'].tolist(),
            "Prediction for Calories": predictions['calories'].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
