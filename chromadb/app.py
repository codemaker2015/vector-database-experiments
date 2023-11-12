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