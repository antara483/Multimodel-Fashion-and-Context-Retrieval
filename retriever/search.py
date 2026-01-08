
import faiss
import numpy as np
import open_clip
import torch
from sklearn.metrics.pairwise import cosine_similarity

from query_parser import parse_query

# Load FAISS index and metadata
clip_index = faiss.read_index("../indexer/clip.index")

# Metadata contains image name, image_id, and attribute IDs
metadata = np.load("../indexer/metadata.npy", allow_pickle=True)
#  Precomputed color histograms for each image
color_embeddings = np.load("../indexer/color_embeddings.npy")

# Load CLIP text encoder
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model.eval()

def encode_text(text):
    """
    Converts a natural language query into a CLIP text embedding.

    Steps:
    1. Tokenize input text
    2. Encode using CLIP text encoder
    3. Normalize embedding for cosine similarity
    """
    tokens = open_clip.tokenize([text])
    with torch.no_grad():
        emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy()

def search(query, k=5):
    """
    Retrieves top-k matching images for a given text query.

    Pipeline:
    1. Encode text using CLIP
    2. Retrieve top candidates from FAISS index
    3. Apply soft re-ranking using context & color cues
    """
    query_emb = encode_text(query)

    # Parse query into weak semantic signals (color, context, clothing)
    parsed = parse_query(query)

    # Get more candidates first
    clip_scores, indices = clip_index.search(
        np.array([query_emb]).astype("float32"), k * 5
    )

    results = []

    for rank, idx in enumerate(indices[0]):
        meta = metadata[idx]

        # 1️⃣ Base similarityscore from CLIP
        final_score = float(clip_scores[0][rank])

        # 2️⃣ Soft context intent boosting
        # (NOT strict filtering)
        if "formal" in parsed["context"]:
            final_score *= 1.1
        if "casual" in parsed["context"]:
            final_score *= 1.1

        # 3️⃣ Soft color consistency boost
        if parsed["colors"]:
            img_color = color_embeddings[idx]
            color_score = np.max(img_color)  # simple heuristic
            final_score += 0.05 * color_score

        results.append((meta["image"], final_score))
    # Sort by final score
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]

# Testing
if __name__ == "__main__":
    query = "A red tie and a white shirt in a formal setting"
    results = search(query, k=5)
    for img, score in results:
        print(img, score)
