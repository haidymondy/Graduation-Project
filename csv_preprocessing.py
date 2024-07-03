import pandas as pd
import numpy as np
import cmath

df = pd.read_csv('nutrients_csvfile (2).csv')
df


df['Grams'] = pd.to_numeric(df['Grams'], errors='coerce')
df['Protein'] = pd.to_numeric(df['Protein'], errors='coerce')
df['Fat'] = pd.to_numeric(df['Fat'], errors='coerce')
df['Carbs'] = pd.to_numeric(df['Carbs'], errors='coerce')
df['Sat.Fat'] = pd.to_numeric(df['Sat.Fat'], errors='coerce')
df['Calories'] = pd.to_numeric(df['Calories'], errors='coerce')
df['Fiber'] = pd.to_numeric(df['Fiber'], errors='coerce')

df['Protein'] = ((df['Protein'] / df['Grams']) * 100).round()
df['Fat'] = ((df['Fat'] / df['Grams']) * 100).round()
df['Carbs'] = ((df['Carbs'] / df['Grams']) * 100).round()

df['Sat.Fat'] = ((df['Sat.Fat'] / df['Grams']) * 100).round()
df['Calories'] = ((df['Calories'] / df['Grams']) * 100).round()
df['Fiber'] = ((df['Fiber'] / df['Grams']) * 100).round()
df['Grams'] = 100
df.to_csv('normalized_data.csv', index=False)


#***************************************************************
import pandas as pd
csv_file_path ='foodcalories (1).csv'   # Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv('foodcalories (1).csv' )
df.head()





# Load the CSV file into a DataFrame
csv_file_path = 'foodcalories (1).csv'  # Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv(csv_file_path)

# Function to extract the numeric part before 'g' from a cell
def extract_numeric_part(cell_value):
    try:
        # Extract numeric part using regex
        numeric_part = int(re.search(r'(\d+) g', cell_value).group(1))
        return numeric_part
    except (AttributeError, ValueError):
        # Handle the case where the quantity is not a valid number or not found
        return None

df['Serving'] = df['Serving'].apply(extract_numeric_part)

df.to_csv('extracted_numeric_file.csv', index=False)
df['Calories'] = df['Calories'].str.replace('cal', '')
df.head()



df['Serving'] = pd.to_numeric(df['Serving'], errors='coerce')
df['Protein (g)'] = pd.to_numeric(df['Protein (g)'], errors='coerce')
df['Fat (g)'] = pd.to_numeric(df['Fat (g)'], errors='coerce')
df['Carbohydrates (g)'] = pd.to_numeric(df['Carbohydrates (g)'], errors='coerce')
df['Calories'] = pd.to_numeric(df['Calories'], errors='coerce')
df.head()




df['Protein (g)'] = ((df['Protein (g)'] / df['Serving']) * 100).round()
df['Fat (g)'] = ((df['Fat (g)'] / df['Serving']) * 100).round()
df['Carbohydrates (g)'] = ((df['Carbohydrates (g)'] / df['Serving']) * 100).round()
df['Calories'] = ((df['Calories'] / df['Serving']) * 100).round()
df['Serving'] = 100

df.head()




df.to_csv('normalized_food_calories.csv', index=False)



