import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("Starting model training...")

# 1. Load Data
df = pd.read_csv("cuisine_dataset.csv")

# Clean up any potential whitespace issues
# *** FIX 1: Updated to use .map to remove the warning ***
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

# 2. Define Features (X) and Target (y)
X = df.drop(["cuisine", "dish"], axis=1)
y = df["dish"]

# 3. Define the exact order of all categories
diet_type_cats = ['Veg', 'Non-Veg']
hunger_cats = ['Light Snack', 'Moderate Meal', 'Very Hungry']
spice_cats = ['No Spice', 'Mild', 'Medium', 'Spicy', 'Very Spicy']
flavor_cats = ['Creamy / Rich', 'Light / Fresh', 'Savory / Umami', 'Sweet', 'Tangy / Sour']
occasion_cats = ['Casual Weeknight', 'Comfort Food', 'Experimental', 'Healthy / Low-Calorie', 'Quick & Easy', 'Special Occasion']
component_cats = [
    'Bread-based', 'Lentils / Beans', 'Mainly Vegetables', 'Noodles / Pasta', 'Paneer / Tofu', 'Rice-based',
    'Chicken', 'Fish', 'Mutton', 'Prawns' # Added all component types
]

all_categories = [diet_type_cats, hunger_cats, spice_cats, flavor_cats, occasion_cats, component_cats]

# 4. Encode Features (X)
feature_encoder = OrdinalEncoder(
    categories=all_categories,
    handle_unknown='use_encoded_value', 
    unknown_value=-1 
)
X_encoded = feature_encoder.fit_transform(X)

# 5. Encode Target (y) - Now encoding DISHES
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- Model Training & Evaluation ---
# *** FIX 2: Removed stratify=y_encoded to fix the ValueError ***
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"--- Model Test Accuracy (predicting dishes): {accuracy * 100:.2f}% ---")
print("(Note: This accuracy is a rough estimate, as the data was not stratified.)")


# --- Retrain on FULL dataset for deployment ---
print("Retraining on full dataset for final model...")
model.fit(X_encoded, y_encoded)
print("Model retrained successfully.")

# 6. Save the final model and encoders
joblib.dump(model, "model.joblib")
joblib.dump(feature_encoder, "feature_encoder.joblib")
joblib.dump(label_encoder, "target_encoder.joblib") # This now encodes dishes

print("\n✅ SUCCESS: All files saved.")