from transformers import CLIPProcessor, CLIPModel
import torch
import faiss
from PIL import Image
import numpy as np

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

corpus = [
    ("image1.jpg", "sample text 1."),
    ("image2.jpg", "sample text 2."),
    ("image3.jpg", "sample text 3.")
]

def get_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
    return image_embedding.cpu().numpy()

def get_text_embedding(text):
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return text_embedding.cpu().numpy()

image_embeddings = np.vstack([get_image_embedding(img) for img, _ in corpus])
text_embeddings = np.vstack([get_text_embedding(txt) for _, txt in corpus])

def build_faiss_index(embeddings, dimension):
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

embedding_dim = image_embeddings.shape[1]

image_index = build_faiss_index(image_embeddings, embedding_dim)
text_index = build_faiss_index(text_embeddings, embedding_dim)

def search_image(query_image_path, k=5):
    query_embedding = get_image_embedding(query_image_path)
    D, I = text_index.search(query_embedding, k)
    return list(zip(I[0], D[0]))

def search_text(query_text, k=5):
    query_embedding = get_text_embedding(query_text)
    D, I = image_index.search(query_embedding, k)
    return list(zip(I[0], D[0]))

def combined_sort(text_results, image_results, w1=0.5, w2=0.5):
    text_rank = {doc: rank for rank, (doc, _) in enumerate(text_results)}
    image_rank = {doc: rank for rank, (doc, _) in enumerate(image_results)}

    all_docs = set(text_rank.keys()) | set(image_rank.keys())

    combined_scores = {}
    for doc in all_docs:
        rank_text = text_rank.get(doc, len(text_results))
        rank_image = image_rank.get(doc, len(image_results))
        score = 0.5 * (1 / (rank_text + 1)) + 0.5 * (1 / (rank_image + 1))
        combined_scores[doc] = score

    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

query_image_path = "query_image.jpg"
query_text = "query text."

image_to_text_results = search_image(query_image_path, k=5)
print(f"Image to Text Search Results: {image_to_text_results}")

text_to_image_results = search_text(query_text, k=5)
print(f"Text to Image Search Results: {text_to_image_results}")

final_results = combined_sort(image_to_text_results, text_to_image_results)
print(f"Combined and Sorted Results: {final_results}")