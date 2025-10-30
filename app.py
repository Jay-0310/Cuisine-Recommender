import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# --- Load Models & Data ---
base_dir = os.path.dirname(os.path.abspath(__file__))

try:
    # Load the trained model and encoders
    model_path = os.path.join(base_dir, "model.joblib")
    encoder_path = os.path.join(base_dir, "feature_encoder.joblib")
    label_encoder_path = os.path.join(base_dir, "target_encoder.joblib")
    
    model = joblib.load(model_path)
    feature_encoder = joblib.load(encoder_path)
    label_encoder = joblib.load(label_encoder_path) # This encodes dishes
    
    # Load the dataset to map dishes back to cuisines
    data_path = os.path.join(base_dir, "cuisine_dataset.csv")
    df = pd.read_csv(data_path)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Create the Dish -> Cuisine mapping
    dish_to_info_map = {}
    
    # Create a map of { cuisine -> [dish_info, dish_info...] }
    cuisine_to_dishes_map = {} 
    
    for _, row in df.iterrows():
        dish = row['dish']
        cuisine = row['cuisine']
        diet = row['diet_type']
        component = row['component']
        
        dish_info = {
            "cuisine": cuisine,
            "diet_type": diet,
            "component": component,
            "dish": dish
        }
        
        dish_to_info_map[dish] = dish_info
        
        # Add to the new cuisine-to-dishes map
        if cuisine not in cuisine_to_dishes_map:
            cuisine_to_dishes_map[cuisine] = []
        
        # Use a check to avoid duplicates from the doubled dataset
        if dish_info not in cuisine_to_dishes_map[cuisine]:
             cuisine_to_dishes_map[cuisine].append(dish_info)

    print("--- Model, encoders, and data maps loaded successfully ---")
except FileNotFoundError:
    print("---!!! MODEL/DATA FILES NOT FOUND! Please run train.py first. ---")
    model = None
except Exception as e:
    print(f"Error loading models: {e}")
    model = None

# --- Helper Function for Descriptions ---
def generate_description(cuisine, inputs):
    """Generates a dynamic description based on the cuisine and user inputs."""
    s = inputs['spice'].lower()
    f = inputs['flavor'].lower()
    c = inputs['component'].lower()

    descriptions = {
        "Punjabi": f"A rich, classic choice, perfect for your {s} and {f} preferences, often featuring {c}.",
        "Indo-Chinese": f"A popular fusion style, matching your desire for {s} spice and {f} flavors.",
        "Italian": f"A comforting and universally loved option, great for {f} flavors and {c}.",
        "Mexican": f"A vibrant and bold choice, known for its {s}, {f} flavors and use of {c}.",
        "Gujarati": f"A light, often {f} and vegetarian choice, great for a light meal.",
        "Mediterranean": f"A fresh, healthy, and {f} option that's highly rated.",
        "Mughlai (Veg)": f"A royal and rich {f} cuisine, perfect for a special occasion.",
        "South Indian": f"A diverse and flavorful choice, ranging from {s} to mild and {f}.",
        "Thai": f"An exciting and aromatic choice, known for balancing {s}, {f}, and sweet flavors.",
        "Japanese (Veg)": f"A clean, {f}, and healthy option, focusing on fresh components.",
        "Chaat": f"The perfect {s}, {f} street food snack for a quick and easy craving.",
        "Korean (Veg)": f"An adventurous, {s} and {f} option for an experimental meal.",
        "Continental": f"A broad and satisfying {f} choice, often with {c} options.",
        "Sandwich": f"The ultimate quick, easy, and customizable option for any time.",
        "Maharashtrian": f"A flavorful and often {s} choice, famous for its unique {f} dishes like {c}." # <-- ADDED
    }
    return descriptions.get(cuisine, f"A great {f} choice featuring {c}.")


# --- Define Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Receives data from the form, predicts top 3, and returns them.
    """
    if not model:
        return jsonify({"error": "Model is not loaded on the server."}), 500
        
    try:
        data = request.get_json()
        
        user_diet = data.get('diet_type')
        user_component = data.get('component')
        
        diet_inputs_to_check = []
        if user_diet == 'Both':
            diet_inputs_to_check.extend(['Veg', 'Non-Veg'])
        else:
            diet_inputs_to_check.append(user_diet)

        cuisine_scores = {}
        cuisine_top_dishes = {}

        all_dish_names = label_encoder.classes_
        
        for diet_input in diet_inputs_to_check:
            # Create the 6-feature input vector for the model
            input_vector = [[
                diet_input,
                data.get('hunger'),
                data.get('spice'),
                data.get('flavor'),
                data.get('occasion'),
                user_component
            ]]
            
            encoded_data = feature_encoder.transform(input_vector)
            all_probs = model.predict_proba(encoded_data)[0]

            # --- *** THIS IS THE CORRECTED LOGIC *** ---
            # Loop through all possible dishes and their scores
            for i, dish_name in enumerate(all_dish_names):
                prob = all_probs[i]
                if prob == 0: # The model is 0% confident, skip it
                    continue
                
                info = dish_to_info_map.get(dish_name)
                if not info:
                    continue 
                
                # *** CRITICAL FIX: ***
                # We only consider dishes that match the user's diet AND component.
                if info['diet_type'] == diet_input and info['component'] == user_component:
                    cuisine = info['cuisine']
                    
                    # Add to total cuisine score
                    cuisine_scores[cuisine] = cuisine_scores.get(cuisine, 0) + prob
                    
                    # Add to dish list for that cuisine
                    if cuisine not in cuisine_top_dishes:
                        cuisine_top_dishes[cuisine] = []
                    cuisine_top_dishes[cuisine].append((dish_name, prob))

        # If no dishes match the component, cuisine_scores will be empty.
        if not cuisine_scores:
             return jsonify({"error": f"No {user_component} dishes found for these criteria. Please try a different combination!"}), 404

        sorted_cuisines = sorted(cuisine_scores.items(), key=lambda item: item[1], reverse=True)
        
        # --- Stricter Back-fill ---
        recommendations = []
        for cuisine_name, total_score in sorted_cuisines[:3]:
            # Get the top MODEL-predicted dishes
            dish_list = cuisine_top_dishes.get(cuisine_name, [])
            sorted_dishes = sorted(dish_list, key=lambda item: item[1], reverse=True)
            
            top_model_dishes = [d[0] for d in sorted_dishes]
            
            final_dishes_set = set(top_model_dishes)
            final_dishes_list = list(top_model_dishes)
            
            # 1. Back-fill: If we have less than 3 dishes, find more.
            #    We ONLY add dishes that match the user's component.
            if len(final_dishes_list) < 3:
                all_cuisine_dishes = cuisine_to_dishes_map.get(cuisine_name, [])
                for dish_info in all_cuisine_dishes:
                    if len(final_dishes_list) >= 3:
                        break
                    
                    # Check diet AND component
                    if (user_diet == 'Both' or user_diet == dish_info['diet_type']) and \
                       (user_component == dish_info['component']) and \
                       (dish_info['dish'] not in final_dishes_set):
                        
                        final_dishes_list.append(dish_info['dish'])
                        final_dishes_set.add(dish_info['dish'])
            
            # *** BUG FIX: ***
            # The second, broader back-fill has been REMOVED.
            # We will no longer add dishes that don't match the component.

            # Format the output
            recommendations.append({
                "cuisine": cuisine_name,
                "score": float(total_score),
                "description": generate_description(cuisine_name, data),
                "dishes": final_dishes_list[:3] # Show only what we found (max 3)
            })
            
        return jsonify({"recommendations": recommendations})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 400

if __name__ == '__main__':
    app.run(debug=True)