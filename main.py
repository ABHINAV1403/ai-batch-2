import os
from dotenv import load_dotenv
from groq import Groq

# Load API keys
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY is missing. Check your .env file.")

# Initialize Groq client
client = Groq(api_key=api_key)

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are an e-commerce competitor analyst."},
        {"role": "user", "content": "Who are you?"},
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)
