import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def generate_synthetic_data(num_samples=1000):
    """
    Generates synthetic data for flood prediction.
    Features:
    - rainfall (mm)
    - river_water_level (m)
    - soil_moisture (%)
    - humidity (%)
    - temperature (Celsius)
    """
    np.random.seed(42)
    
    rainfall = np.random.uniform(0, 300, num_samples)
    river_water_level = np.random.uniform(2, 10, num_samples)
    soil_moisture = np.random.uniform(10, 100, num_samples)
    humidity = np.random.uniform(30, 100, num_samples)
    temperature = np.random.uniform(10, 40, num_samples)
    
    # Calculate a composite "risk score" to determine if a flood occurs
    # Higher rainfall, water level, and soil moisture increase flood risk
    risk_score = (rainfall * 0.4) + (river_water_level * 10) + (soil_moisture * 0.3)
    
    # Add some noise to the risk score
    risk_score += np.random.normal(0, 10, num_samples)
    
    # Define a threshold for flooding
    threshold = 150
    flood_occurrence = (risk_score > threshold).astype(int)
    
    data = pd.DataFrame({
        'Rainfall_mm': rainfall,
        'River_Water_Level_m': river_water_level,
        'Soil_Moisture_Percent': soil_moisture,
        'Humidity_Percent': humidity,
        'Temperature_Celsius': temperature,
        'Flood_Occurrence': flood_occurrence
    })
    
    return data

def main():
    print("=========================================================")
    print(" A Machine Learning Approach for Flood Prediction (Python) ")
    print("=========================================================\n")
    
    print("1. Generating synthetic dataset...")
    df = generate_synthetic_data(num_samples=2000)
    print(f"Dataset generated with {df.shape[0]} samples and {df.shape[1]} columns.")
    print("\nClass distribution (1 = Flood, 0 = No Flood):")
    print(df['Flood_Occurrence'].value_counts().to_string())
    
    print("\n---------------------------------------------------------")
    print("2. Data Preprocessing...")
    X = df.drop('Flood_Occurrence', axis=1)
    y = df['Flood_Occurrence']
    
    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled using StandardScaler.")
    
    print("\n---------------------------------------------------------")
    print("3. Training the Model (Random Forest Classifier)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    print("Model training completed.")
    
    print("\n---------------------------------------------------------")
    print("4. Model Evaluation...")
    y_pred = clf.predict(X_test_scaled)
    
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n---------------------------------------------------------")
    print("5. Feature Importance...")
    feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(feature_importances.to_string())
    
    print("\n---------------------------------------------------------")
    print("6. Predicting on new unseen scenarios...")
    # Sample 1: High risk (High rainfall, high water level)
    # Sample 2: Low risk (Low rainfall, normal water level)
    new_data = pd.DataFrame({
        'Rainfall_mm': [250.0, 20.0],
        'River_Water_Level_m': [8.5, 3.0],
        'Soil_Moisture_Percent': [90.0, 40.0],
        'Humidity_Percent': [85.0, 50.0],
        'Temperature_Celsius': [25.0, 30.0]
    })
    
    new_data_scaled = scaler.transform(new_data)
    predictions = clf.predict(new_data_scaled)
    prediction_probs = clf.predict_proba(new_data_scaled)
    
    for i, pred in enumerate(predictions):
        status = "CRITICAL: Flood Likely" if pred == 1 else "SAFE: No Flood"
        prob = prediction_probs[i][1] if pred == 1 else prediction_probs[i][0]
        print(f"Scenario {i+1} -> {status} (Confidence: {prob*100:.2f}%)")
    print("=========================================================")

if __name__ == "__main__":
    main()
