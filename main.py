import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
from ultralytics import YOLO
import numpy as np
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import io
import pandas as pd
from typing import Dict, List, Any

model = YOLO(r"G:\last year\GP\best (8).pt")
app = FastAPI()

df = pd.read_csv('final_calories_1 (2).csv')
df2 = pd.read_csv(r'normalized_ingredients (2).csv')


class FoodItem(BaseModel):
    name: str
    quantity: float


class FoodSearch(BaseModel):
    input: str


class Item(BaseModel):
    Height: float
    Weight: float
    Age: int
    Gender: str


class NutritionDetails(BaseModel):
    total_calories: float
    total_protein: float
    total_fat: float
    total_carbohydrates: float
    healthy: bool


class IngredientDetails(BaseModel):
    total_calories: float
    total_protein: float
    total_fat: float
    total_saturated_fat: float
    total_carbohydrates: float


class FoodElements(BaseModel):
    bmr: float
    elements: Dict[str, float]


def img_detect(img):
    detect_result = model(img)
    detect_img = detect_result[0].plot()
    _, img_encoded = cv2.imencode(".jpg", detect_img)

    class_ids = detect_result[0].boxes.cls
    class_names = [model.names[int(cls_id)] for cls_id in class_ids]
    unique = list(set(class_names))
    return img_encoded.tobytes(), unique


def calculate_macronutrients(bmr):
    calories_per_gram = {
        'Carbohydrates': 4,
        'Fat': 9,
        'Protein': 4
    }

    percentage_ranges = {
        'Carbohydrates': (0.45, 0.55),
        'Fat': (0.30, 0.35),
        'Protein': (0.15, 0.20)
    }

    macronutrient_grams = {}
    for nutrient, range_ in percentage_ranges.items():
        lower_range_calories = bmr * range_[0]
        upper_range_calories = bmr * range_[1]

        lower_range_grams = lower_range_calories / calories_per_gram[nutrient]
        upper_range_grams = upper_range_calories / calories_per_gram[nutrient]

        macronutrient_grams[nutrient] = (lower_range_grams, upper_range_grams)

    return macronutrient_grams


@app.post("/food")
async def upload_food_elements(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    _, class_names = img_detect(img)
    return {"class_names": class_names}


@app.post("/image")
async def upload_image(image: UploadFile = File(...)):
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    detected_img_bytes, _ = img_detect(img)
    image_stream = io.BytesIO(detected_img_bytes)

    # Return the image as a streaming response
    return StreamingResponse(image_stream, media_type="image/jpeg")


@app.post("/items")
async def create_item(item: Item):
    Weight = item.Weight
    Height = item.Height
    Age = item.Age
    Gender = item.Gender

    if Gender.lower() == 'male':
        bmr = 88.362 + (13.397 * Weight) + (4.799 * Height) - (5.677 * Age)
        print("bmr for male", bmr)
    elif Gender.lower() == 'female':
        bmr = 447.593 + (9.247 * Weight) + (3.098 * Height) - (4.330 * Age)
        print("bmr for female", bmr)

    else:
        raise HTTPException(status_code=400, detail="Invalid gender. Please specify 'male' or 'female'.")

    return {"bmr": bmr}


@app.post("/details")
async def calculate_nutrition(elements: FoodElements):
    bmr = elements.bmr
    my_map = elements.elements
    food_map = {}
    total_calories = 0
    total_protein = 0
    total_fat = 0
    total_carbohydrates = 0

    for food, value in my_map.items():
        # Assuming you have filtered_df and df initialized earlier
        filtered_df = df[df['Food'] == food]

        # Check if the filtered DataFrame is not empty
        if not filtered_df.empty:
            # Extract nutritional information and multiply each value by its corresponding value in my_map
            calories = filtered_df['Calories'].values[0] * (value / 100)
            protein = filtered_df['Protein (g)'].values[0] * (value / 100)
            fat = filtered_df['Fat (g)'].values[0] * (value / 100)
            carbohydrates = filtered_df['Carbohydrates (g)'].values[0] * (value / 100)

            # Add adjusted nutritional values to the totals
            total_calories += calories
            total_protein += protein
            total_fat += fat
            total_carbohydrates += carbohydrates

            food_map[food] = {'Calories': calories, 'Protein (g)': protein, 'Fat (g)': fat,
                              'Carbohydrates (g)': carbohydrates}

        else:
            food_map[food] = {'message': f"No nutritional information found for '{food}'."}

    macronutrient_grams = calculate_macronutrients(bmr)
    for nutrient, grams_range in macronutrient_grams.items():
        print(f"{nutrient}: {grams_range[0]:.2f}g - {grams_range[1]:.2f}g")

    healthy = (total_calories < bmr and
               total_carbohydrates <= macronutrient_grams['Carbohydrates'][1] and
               total_protein <= macronutrient_grams['Protein'][1] and
               total_fat <= macronutrient_grams['Fat'][1])

    return NutritionDetails(
        total_calories=total_calories,
        total_protein=total_protein,
        total_fat=total_fat,
        total_carbohydrates=total_carbohydrates,
        healthy=healthy
    )


@app.post('/search_food')
async def search_food(data: FoodSearch):
    try:
        input_text = data.input.strip().lower()

        if not input_text:
            return []

        filtered_df = df2[df2['Food'].str.lower().str.contains(input_text)]

        if filtered_df.empty:
            return []

        matching_food_names = filtered_df['Food'].tolist()

        return matching_food_names

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/calculate_nutrition')
async def calculate_nutrition(data: FoodItem):
    food_name = data.name.strip().lower()
    quantity = data.quantity

    if not food_name or quantity <= 0:
        raise HTTPException(status_code=400, detail="Invalid input. Please provide a valid food name and quantity.")

    filtered_df = df2[df2['Food'].str.lower() == food_name]

    if not filtered_df.empty:
        selected_food_info = filtered_df.iloc[0]
        grams_per_quantity = selected_food_info['Grams']
        calories_per_gram = selected_food_info['Calories'] / grams_per_quantity
        protein_per_gram = selected_food_info['Protein'] / grams_per_quantity
        fat_per_gram = selected_food_info['Fat'] / grams_per_quantity
        sat_fat_per_gram = selected_food_info['Sat.Fat'] / grams_per_quantity
        carbs_per_gram = selected_food_info['Carbs'] / grams_per_quantity

        # Replace NaN values with zeros
        calories_per_gram = np.nan_to_num(calories_per_gram)
        protein_per_gram = np.nan_to_num(protein_per_gram)
        fat_per_gram = np.nan_to_num(fat_per_gram)
        sat_fat_per_gram = np.nan_to_num(sat_fat_per_gram)
        carbs_per_gram = np.nan_to_num(carbs_per_gram)

        calories = calories_per_gram * quantity
        protein = protein_per_gram * quantity
        fat = fat_per_gram * quantity
        sat_fat = sat_fat_per_gram * quantity
        carbs = carbs_per_gram * quantity

        return {"calories": calories,
                "protein": protein,
                "fat": fat,
                "sat_fat": sat_fat,
                "carbs": carbs}
    else:
        raise HTTPException(status_code=404, detail=f"'{food_name.capitalize()}' not found in the food list.")


@app.post('/calculate_total_nutrition')
async def calculate_total_nutrition(body: Dict[str, List[Dict[str, Any]]]):
    try:
        total_calories = 0
        total_protein = 0
        total_fat = 0
        total_sat_fat = 0
        total_carbs = 0

        items = body.get("total", [])

        for item in items:
            food_name = item.get("name", "").strip().lower()
            quantity = item.get("quantity", 0)

            filtered_df = df2[df2['Food'].str.lower() == food_name]

            if not filtered_df.empty:
                selected_food_info = filtered_df.iloc[0]
                grams_per_quantity = selected_food_info['Grams']
                calories_per_gram = selected_food_info['Calories'] / grams_per_quantity
                protein_per_gram = selected_food_info['Protein'] / grams_per_quantity
                fat_per_gram = selected_food_info['Fat'] / grams_per_quantity
                sat_fat_per_gram = selected_food_info['Sat.Fat'] / grams_per_quantity
                carbs_per_gram = selected_food_info['Carbs'] / grams_per_quantity

                # Replace NaN values with zeros
                calories_per_gram = np.nan_to_num(calories_per_gram)
                protein_per_gram = np.nan_to_num(protein_per_gram)
                fat_per_gram = np.nan_to_num(fat_per_gram)
                sat_fat_per_gram = np.nan_to_num(sat_fat_per_gram)
                carbs_per_gram = np.nan_to_num(carbs_per_gram)

                calories = calories_per_gram * quantity
                protein = protein_per_gram * quantity
                fat = fat_per_gram * quantity
                sat_fat = sat_fat_per_gram * quantity
                carbs = carbs_per_gram * quantity

                total_calories += calories
                total_protein += protein
                total_fat += fat
                total_sat_fat += sat_fat
                total_carbs += carbs

        return IngredientDetails(
            total_calories=total_calories,
            total_protein=total_protein,
            total_fat=total_fat,
            total_saturated_fat=total_sat_fat,
            total_carbohydrates=total_carbs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, port=5000, host='localhost')
