# Vector Databases

A vector database is a type of database that stores data as high-dimensional vectors, which are mathematical representations of features or attributes. Unlike traditional relational databases with rows and columns, data points in a vector database are represented by vectors with a fixed number of dimensions, clustered based on similarity. This design enables low latency queries, making them ideal for AI-driven applications 

In this repository, we'll be experimenting on various vector databases such as Chromadb, Pinecone, Weaviate and Pgvector.

## Chromadb

#### Installation

```
pip install chromadb==0.3.29 openai==0.28.1 wget numpy pandas
```

#### Code

```py
import openai
import pandas as pd
import os
import wget
import zipfile
from ast import literal_eval

# Chroma's client library for Python
import chromadb

# I've set this to our new embeddings model, this can be changed to the embedding model of your choice
EMBEDDING_MODEL = "text-embedding-ada-002"

# Ignore unclosed SSL socket warnings - optional in case you get these errors
import warnings

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'

# The file is ~700 MB so this will take some time
wget.download(embeddings_url)

with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip","r") as zip_ref:
    zip_ref.extractall("data")

article_df = pd.read_csv('data/vector_database_wikipedia_articles_embedded.csv')

print(article_df.head())


# Read vectors from strings back into a list
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Set vector_id to be a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

print(article_df.info(show_counts=True))


# Instantiate the Chroma client
# Create the Chroma client. By default, Chroma is ephemeral and runs in memory.

chroma_client = chromadb.Client()

# Create collections
# Chroma collections allow you to store and filter with arbitrary metadata, making it easy to query subsets of the embedded data.

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

os.environ["OPENAI_API_KEY"] = 'openai-api-key'
if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")


embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=EMBEDDING_MODEL)

wikipedia_content_collection = chroma_client.create_collection(name='wikipedia_content', embedding_function=embedding_function)
wikipedia_title_collection = chroma_client.create_collection(name='wikipedia_titles', embedding_function=embedding_function)


# Fill the collections
# Chroma collections allow you to populate, and filter on, whatever metadata you like.

# Add the content vectors
wikipedia_content_collection.add(
    ids=article_df.vector_id.tolist(),
    embeddings=article_df.content_vector.tolist(),
)

# Add the title vectors
wikipedia_title_collection.add(
    ids=article_df.vector_id.tolist(),
    embeddings=article_df.title_vector.tolist(),
)


# Search the collections
# Chroma handles embedding queries for you if an embedding function is set, like in this example.

def query_collection(collection, query, max_results, dataframe):
    results = collection.query(query_texts=query, n_results=max_results, include=['distances']) 
    df = pd.DataFrame({
                'id':results['ids'][0], 
                'score':results['distances'][0],
                'title': dataframe[dataframe.vector_id.isin(results['ids'][0])]['title'],
                'content': dataframe[dataframe.vector_id.isin(results['ids'][0])]['text'],
                })
    
    return df

title_query_result = query_collection(
    collection=wikipedia_title_collection,
    query="modern art in India",
    max_results=10,
    dataframe=article_df
)

print(title_query_result.head())


content_query_result = query_collection(
    collection=wikipedia_content_collection,
    query="Famous battles in Indian history",
    max_results=10,
    dataframe=article_df
)
print(content_query_result.head())
```

#### Screenshots 

[screenshots](screenshots/chromadb-output.png)

## Pinecone

```
pip install pinecone-client
```

#### Code

```py
import pinecone

# Initialize your connection
# Using your API key and environment, initialize your client connection to Pinecone:
pinecone.init(api_key="xxxxxxxxxxxx", environment="xxxxxxxxxxxx")

# Create an index
# Create an index named "quickstart" that performs nearest-neighbor search using the Euclidean distance metric for 8-dimensional vectors:
pinecone.create_index("quickstart", dimension=8, metric="euclidean")
pinecone.describe_index("quickstart")

# Insert vectors
# Use the upsert operation to write 5 8-dimensional vectors into the index:
index = pinecone.Index("quickstart")
index.upsert(
  vectors=[
    {"id": "A", "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
    {"id": "B", "values": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]},
    {"id": "C", "values": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]},
    {"id": "D", "values": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]},
    {"id": "E", "values": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}
  ]
)

# Run a nearest-neighbor search
res = index.query(
  vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
  top_k=3,
  include_values=True
)
print(res)

# Clean up
# In Pinecone's Starter plan, only one index is allowed. To remove the "quickstart" index, use the delete_index operation.
pinecone.delete_index("quickstart")
```

#### Screenshots 

[screenshots](screenshots/pinecone-output.png)

## Weaviate

#### Installation
```
pip install "weaviate-client==3.*"
```

#### Code

```py
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
```

#### Screenshots 

[screenshots](screenshots/weaviate-output.png)

## Pgvector

```
pip install pgvector psycopg openai==0.28.1 psycopg-binary
```

#### Code

```py
import openai
from pgvector.psycopg import register_vector
import psycopg

conn = psycopg.connect(
    host='localhost',
    port=5432,
    user='postgres',
    password='123456',
    dbname='pgvector_example',
    autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(1536))')

input = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling',
    'The bird is singing',
    'The elephant is trumpeting',
    'The lion is roaring',
    'The horse is neighing',
    'The monkey is chattering'
]

import os
os.environ["OPENAI_API_KEY"] = 'your-openai-api-key'
if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")

response = openai.Embedding.create(input=input, model='text-embedding-ada-002')
embeddings = [v['embedding'] for v in response['data']]

for content, embedding in zip(input, embeddings):
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, embedding))

document_id = 1
neighbors = conn.execute('SELECT content FROM documents WHERE id != %(id)s ORDER BY embedding <=> (SELECT embedding FROM documents WHERE id = %(id)s) LIMIT 5', {'id': document_id}).fetchall()
for neighbor in neighbors:
    print(neighbor[0])
```

#### Screenshots 

[screenshots](screenshots/pgvector-output.png)