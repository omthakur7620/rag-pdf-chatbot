# scripts/list_groq_models.py

from dotenv import load_dotenv
import os
from groq import Groq

# Load environment variables
load_dotenv()

def list_models():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    models = client.models.list()

    print("\n✅ Available Groq Models:\n")
    for model in models.data:
        print("-", model.id)

if __name__ == "__main__":
    list_models()
