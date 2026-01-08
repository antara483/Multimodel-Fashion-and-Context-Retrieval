
import os
import matplotlib.pyplot as plt
from PIL import Image

from search import search
# Directory containing indexed images
IMAGE_DIR = "../data/images"
# Evaluation queries (provided)
evaluation_queries = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting."
]


# Run retrieval & visualize results
for query in evaluation_queries:
    results = search(query, k=5)

    plt.figure(figsize=(15, 5))

    for i, (img_name, _) in enumerate(results):
        img_path = os.path.join(IMAGE_DIR, img_name)

        # Load and display image
        img = Image.open(img_path).convert("RGB")

        plt.subplot(1, len(results), i + 1)
        plt.imshow(img)
        plt.title(f"Top {i+1}")
        plt.axis("off")
    # Display query as title
    plt.suptitle(query, fontsize=14)
    plt.show()
