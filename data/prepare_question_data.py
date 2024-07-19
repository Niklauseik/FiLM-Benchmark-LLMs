import os
import pandas as pd
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ChanceFocus/flare-finqa", split='test')

# Convert the dataset to a DataFrame and select the necessary columns
df = pd.DataFrame(dataset)
df = df[['query', 'answer']]

# Save the DataFrame to a CSV file
csv_path = 'data/question_data.csv'
df.to_csv(csv_path, index=False)
print(f"Dataset saved to {csv_path}")
