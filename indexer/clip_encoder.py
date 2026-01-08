import torch
import open_clip
from PIL import Image


# -----------------------------------
# Load CLIP model and preprocessing
# -----------------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model.eval()

def encode_image(image_path):
    """
    Encodes an image into a CLIP embedding.

    Steps:
    1. Load image
    2. Apply CLIP preprocessing
    3. Encode using CLIP image encoder
    4. Normalize embedding for cosine similarity
    """
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(image)
     # Normalize embedding    
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy()
