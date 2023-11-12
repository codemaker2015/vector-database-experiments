import pinecone

# Initialize your connection
# Using your API key and environment, initialize your client connection to Pinecone:
pinecone.init(api_key="xxxxxxxxxxxxxxxx", environment="xxxxxxxxxxxxxx")

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

