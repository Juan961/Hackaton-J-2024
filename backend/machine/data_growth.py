import joblib
from sklearn.ensemble import RandomForestClassifier


from openaiclient import generate_response


FEATURES_REQUIRED = ["Soil_Type", "Sunlight_Hours", "Water_Frequency", "Fertilizer_Type", "Temperature", "Humidity"]


PROMPT = """
I have done an analysis of a data in the plant and determined that the plant is growing: {growing}.
The features that were used in the analysis are:
- Soil Type: {soil_type}
- Sunlight Hours: {sunlight_hours}
- Water Frequency: {water_frequency}
- Fertilizer Type: {fertilizer_type}
- Temperature: {temperature}
- Humidity: {humidity}
Generate a short response [70 - 100 words] based on the result of my analisys.
"""


def predict_classification(data:dict):
    if len(data) != len(FEATURES_REQUIRED):
        raise ValueError("Invalid data longitude")

    if not all([feature in data for feature in FEATURES_REQUIRED]):
        raise ValueError("Invalid data keys")

    x = [
        data["Sunlight_Hours"], # Sunlight_Hours
        data["Temperature"], # Temperature
        data["Humidity"], # Humidity
        1.00 if data["Soil_Type"] == "clay" else 0.00, # Soil_Type_clay
        1.00 if data["Soil_Type"] == "loam" else 0.00, # Soil_Type_loam
        1.00 if data["Soil_Type"] == "sandy" else 0.00, # Soil_Type_sandy
        1.00 if data["Water_Frequency"] == "bi-weekly" else 0.00, # Water_Frequency_bi-weekly
        1.00 if data["Water_Frequency"] == "daily" else 0.00, # Water_Frequency_daily
        1.00 if data["Water_Frequency"] == "weekly" else 0.00, # Water_Frequency_weekly
        1.00 if data["Fertilizer_Type"] == "chemical" else 0.00, # Fertilizer_Type_chemical
        1.00 if data["Fertilizer_Type"] == "none" else 0.00, # Fertilizer_Type_none
        1.00 if data["Fertilizer_Type"] == "organic" else 0.00, # Fertilizer_Type_organic
    ]

    loaded_model = joblib.load('./machine/models/random_forest_model.pkl')

    prediction = loaded_model.predict([x])

    is_growing = prediction.tolist()[0] == 1

    response = generate_response(
        PROMPT.replace("{growing}", str(is_growing))
        .replace("{soil_type}", str(data["Soil_Type"]))
        .replace("{sunlight_hours}", str(data["Sunlight_Hours"]))
        .replace("{water_frequency}", str(data["Water_Frequency"]))
        .replace("{fertilizer_type}", str(data["Fertilizer_Type"]))
        .replace("{temperature}", str(data["Temperature"]))
        .replace("{humidity}", str(data["Humidity"]))
    )

    return {
        "response": response,
        "growing": is_growing
    }
