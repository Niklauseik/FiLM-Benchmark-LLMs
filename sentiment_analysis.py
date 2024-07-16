import openai
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key

# Load the processed dataset
dataset_path = 'data/sentiment.csv'
dataset = pd.read_csv(dataset_path)

test_dataset = dataset

def analyze_sentiment(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": query}
        ]
    )
    sentiment_analysis = response['choices'][0]['message']['content'].strip().split()[0].lower()  
    return sentiment_analysis

# Add a column for predictions
test_dataset['predicted_sentiment'] = test_dataset['query'].apply(analyze_sentiment)

# Convert the actual answers to lowercase
test_dataset['answer'] = test_dataset['answer'].str.lower()

# Calculate Accuracy and F1 Score
accuracy = accuracy_score(test_dataset['answer'], test_dataset['predicted_sentiment'])
f1 = f1_score(test_dataset['answer'], test_dataset['predicted_sentiment'], average='weighted')

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")

# Save the results to a new CSV file
output_path = 'data/sentiment_analysis_test_results.csv'
test_dataset.to_csv(output_path, index=False)
print(f"Test results saved to {output_path}")