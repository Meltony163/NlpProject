from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector,Prefetch,FusionQuery,Fusion
from dotenv import load_dotenv
from pathlib import Path
import os

dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

qdrant_api_key = os.getenv("qdrant_api_key")
qdrant_url = os.getenv("qdrant_url")

dense_model = SentenceTransformer("all-MiniLM-L6-v2")
sparse_model = SparseTextEmbedding("Qdrant/bm25")

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key)

def CheckQuery(query):
  if isinstance(query, str):
      return True

  try:
      return str(query)
  except :
      return False

def retreive(query):
  if not CheckQuery(query):
    return -1

  query_dense  = dense_model.encode(query).tolist()
  query_sparse = list(sparse_model.embed([query]))[0]

  results = client.query_points(
      collection_name="VectorDatabase",
      prefetch=[
          Prefetch(query=query_dense, using="dense", limit=10),
          Prefetch(
              query=SparseVector(
                  indices=query_sparse.indices.tolist(),
                  values=query_sparse.values.tolist(),
              ),
              using="sparse",
              limit=10,
          ),
      ],
      query=FusionQuery(fusion=Fusion.RRF),
      limit=5,
      with_payload=True,
  )

  results_list = []
  for point in getattr(results, 'points', []):
    payload = getattr(point, 'payload', {}) or {}
    context = payload.get('Context') or payload.get('context')
    response = payload.get('Response') or payload.get('response')
    if context is not None and response is not None:
      results_list.append((context, response))

  return results_list


if __name__ == "__main__":
  user_message = input("User: ")
  res = retreive(user_message)
  for message in res:
    print(message[0])