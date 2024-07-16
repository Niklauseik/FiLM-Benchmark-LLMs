import openai
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("API key not found. Please set it in the .env file.")

openai.api_key = api_key

def analyze_sentiment(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes the sentiment of financial posts."},
            {"role": "user", "content": f"Analyze the sentiment of the following financial post and categorize it as Positive, Negative, or Neutral: {text}"}
        ]
    )
    # Print the response for debugging purposes
    print(response)
    # Extract the message content from the response
    sentiment_analysis = response['choices'][0]['message']['content'].strip()
    return sentiment_analysis

# Example usage
financial_post = "What's up with $LULU? Numbers looked good, not great, but good. I think conference call will instill confidence."
sentiment = analyze_sentiment(financial_post)
print(f"The sentiment of the post is: {sentiment}")
