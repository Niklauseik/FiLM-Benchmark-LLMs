import openai
import os
import pandas as pd
import re
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("API key not found. Please set it in the .env file.")

openai.api_key = api_key

# Load the processed dataset
dataset_path = 'data/question_data.csv'
dataset = pd.read_csv(dataset_path)

def answer_question(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": query}
        ]
    )
    answer = response['choices'][0]['message']['content'].strip()  # Get the model's response
    return answer

# Function to extract numeric values from a string, including negative numbers
def extract_numbers(text):
    return [abs(float(num)) for num in re.findall(r'-?\d*\.\d+|-?\d+', text)]

# Function to normalize answers for comparison
def normalize_answer(answer):
    return answer.lower().strip()

# Enhanced function to check if the correct answer appears in the prediction
def is_correct(pred, true):
    normalized_true = normalize_answer(true)

    # Check for yes/no answers before normalization
    if normalized_true in ["yes", "no"]:
        normalized_pred = normalize_answer(pred)
        return normalized_true in normalized_pred

    # Normalize predicted answer after checking yes/no
    normalized_pred = normalize_answer(pred)

    try:
        # Extract numbers and compare
        pred_nums = extract_numbers(normalized_pred)
        true_nums = extract_numbers(normalized_true)

        if true_nums:
            true_num = true_nums[0]  # Process true answer directly
            for pred_num in pred_nums:
                # Handle percentage comparison
                true_num_as_percent = true_num * 100 if true_num < 1 and true_num > -1 else true_num
                pred_num_as_percent = pred_num * 100 if pred_num < 1 and pred_num > -1 else pred_num
                tolerance = 0.01 * max(abs(pred_num_as_percent), abs(true_num_as_percent))

                if abs(pred_num_as_percent - true_num_as_percent) <= tolerance:
                    return True

                tolerance = 0.01 * max(abs(pred_num), abs(true_num))
                if abs(pred_num - true_num) <= tolerance:
                    return True
            return False
        else:
            return normalized_true in normalized_pred
    except ValueError:
        return normalized_true in normalized_pred

# Add a column for predictions and correctness
dataset['predicted_answer'] = dataset.apply(lambda row: answer_question(row['query']), axis=1)
dataset['is_correct'] = dataset.apply(lambda row: is_correct(row['predicted_answer'], row['answer']), axis=1)

# Calculate accuracy
accuracy = dataset['is_correct'].mean()

# Save the results to a new CSV file
output_path = 'data/question_answering_test_results.csv'
dataset.to_csv(output_path, index=False)

# Print the accuracy
print(f"Accuracy: {accuracy}")
