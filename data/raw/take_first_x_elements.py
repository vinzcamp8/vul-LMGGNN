import json
import os
import pandas as pd

# Define the path to the dataset.json file
file_path = os.path.join(os.path.dirname(__file__), 'dataset.json')

# Function to parse the JSON file
def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Parse the dataset.json file
data = parse_json(file_path)

# Convert the JSON data to a DataFrame
df = pd.DataFrame(data)

# Print the keys of the DataFrame
print(df.keys())

# Take only the first 10 elements of the DataFrame
df_first_10 = df.head(100)

# Define the path to the new JSON file
new_file_path = os.path.join(os.path.dirname(__file__), 'first_100_elements.json')

# Save the first 10 elements to the new JSON file
# df_first_10.to_json(new_file_path, orient='records', lines=True)

# Save the first 10 elements to the new JSON file with the same structure as the initial file
df_first_10.to_json(new_file_path, orient='records', indent=4)