

import os
import faiss
import numpy as np

from clip_encoder import encode_image
from color_extractor import extract_color_vector
from attribute_encoder import load_fashionpedia_annotations

print("🚀 build_index.py started")

# Paths to image directory and annotations

IMAGE_DIR = r"D:\Fashion Retrieval\data\images"
ANNOTATION_PATH = r"D:\Fashion Retrieval\data\annotations.json"

# Sanity check: ensure image directory exists
if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

image_files = os.listdir(IMAGE_DIR)
print(f" Total files found in images folder: {len(image_files)}")

# Load Fashionpedia annotations
file_to_image_id, image_id_to_attrs = load_fashionpedia_annotations(ANNOTATION_PATH)
print(f" Loaded annotations for {len(file_to_image_id)} images")

# Containers for embeddings and metadata
clip_embeddings = []
color_embeddings = []
metadata = []

processed = 0

# Iterate through images and encode them
for img_name in image_files:
    #  Handle all common image extensions
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)

    # Map filename → image_id (may be None, if not found in annotations)
    image_id = file_to_image_id.get(img_name)
    
    # Retrieve attribute IDs (used as weak semantic signal)
    attributes = image_id_to_attrs.get(image_id, [])

    # Encode image using CLIP
    clip_emb = encode_image(img_path)

    # Extract color histogram
    color_emb = extract_color_vector(img_path)

    clip_embeddings.append(clip_emb)
    color_embeddings.append(color_emb)

    metadata.append({
        "image": img_name,
        "image_id": image_id,
        "attributes": attributes
    })

    processed += 1
    if processed % 50 == 0:
        print(f"✅ Processed {processed} images")

#  Final checks
if processed == 0:
    raise RuntimeError("❌ No images were processed. Check image extensions or path.")

print(f" Total images indexed: {processed}")

#Convert lists to numpy
clip_embeddings = np.array(clip_embeddings).astype("float32")
color_embeddings = np.array(color_embeddings).astype("float32")

# Build FAISS index (Inner Product = cosine similarity)
index = faiss.IndexFlatIP(clip_embeddings.shape[1])
index.add(clip_embeddings)

#  Save artifacts
faiss.write_index(index, "clip.index")
np.save("metadata.npy", metadata)
np.save("color_embeddings.npy", color_embeddings)

print("🎉 Indexing completed successfully")
