import weaviate
import json
import requests

# Connect to Weaviate
client = weaviate.Client(
    url = "https://my-first-weaviate-cluster-xxxxxx.weaviate.network",  # Replace with your endpoint
    auth_client_secret=weaviate.AuthApiKey(api_key="xxxxxxxxxxxx"),  # Replace w/ your Weaviate instance API key
    additional_headers = {
        "X-OpenAI-Api-Key": "xxxxxxxxxxxxxxxx"  # Replace with your inference API key
    }
)

print(client.schema.get())

# Define a class
class_obj = {
    "class": "Question",
    "vectorizer": "text2vec-openai",  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
    "moduleConfig": {
        "text2vec-openai": {},
        "generative-openai": {}  # Ensure the `generative-openai` module is used for generative queries
    }
}

client.schema.create_class(class_obj)


# Add objects
fname = "jeopardy_tiny_with_vectors_all-OpenAI-ada-002.json"  # This file includes pre-generated vectors
url = f'https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/{fname}'
resp = requests.get(url)
data = json.loads(resp.text)  # Load data

client.batch.configure(batch_size=100)  # Configure batch
with client.batch as batch:  # Configure a batch process
    for i, d in enumerate(data):  # Batch import all Questions
        print(f"importing question: {i+1}")
        properties = {
            "answer": d["Answer"],
            "question": d["Question"],
            "category": d["Category"],
        }
        batch.add_data_object(
            data_object=properties,
            class_name="Question",
            vector=d["vector"]  # Add custom vector
        )


# Semantic search
response = (
    client.query
    .get("Question", ["question", "answer", "category"])
    .with_near_text({"concepts": ["biology"]})
    .with_limit(2)
    .do()
)

print(json.dumps(response, indent=4))