import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load your dataset
df = pd.read_excel("energy_dataset_1.xlsx")

# Convert rows into readable text
documents = []
for _, row in df.iterrows():
    text = (
        f"User used appliance with power rating {row['power_rating_watt']}W "
        f"for {row['daily_usage_hours']} hours resulting in "
        f"{row['daily_energy_kwh']} kWh energy consumption."
    )
    documents.append(text)

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def retrieve_context(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    retrieved_docs = [documents[i] for i in indices[0]]
    return "\n".join(retrieved_docs)